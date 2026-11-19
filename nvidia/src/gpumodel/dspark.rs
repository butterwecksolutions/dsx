//! DSpark, the checkpoint's speculative draft stack (`mtp.*`), on the main GPU; the CPU reference is core/src/dspark.rs
//! and this follows it operation by operation with the decode path's kernels.
//!
//! Three window-only layers (attention over the main model's projected residual of the last `window` positions plus
//! the block, bidirectionally) with their own 128-expert MoE (top-3, checkpoint FP4, `gemv_fp4`) draft `block` tokens
//! at once; the main model's head plus a Markov bias from the previous drafted token finish them in order.
//! `observe` keeps the draft layers' window rings current from `GpuModel::capture_hidden`'s residuals.

use super::{f32buf, DMat, GpuModel};
use crate::gguf::GType;
use crate::gpu::{Arg, Buf, Gpu};
use crate::model::{Mat, Model};
use crate::safetensors::SafeTensors;
use anyhow::{Context, Result};
use std::path::Path;

struct DLayer {
    attn_norm: Buf,
    ffn_norm: Buf,
    q_a: DMat,
    q_a_norm: Buf,
    q_b: DMat,
    kv: DMat,
    kv_norm: Buf,
    o_a: DMat,
    o_b: DMat,
    sinks: Buf,
    hc_attn: (DMat, Buf, Buf),
    hc_ffn: (DMat, Buf, Buf),
    gate_w: DMat,
    gate_bias: Buf,
    /// [expert] (gate w1, up w3, down w2), FP4
    experts: Vec<[DMat; 3]>,
    sh: [DMat; 3],
    /// bf16 [window][head_dim]: the main model's projected residual, post-RoPE, FP8 fake-quantized
    ring: Buf,
}

/// Scratch for one draft (b = block rows)
struct Bufs {
    h: Buf,       // [b][hc][d]
    x: Buf,       // [b][d]
    mix_a: Buf,   // [b][24]
    mix_f: Buf,   // [b][24]
    qr: Buf,      // [b][q_lora]
    q: Buf,       // [b][nh*hd]
    kv: Buf,      // [b][hd] f32
    blk: Buf,     // [b][hd] bf16
    o: Buf,       // [b][nh*hd]
    r: Buf,       // [b][o_groups*o_lora]
    a: Buf,       // [b][d] attention out / MoE out
    sh_out: Buf,  // [b][d]
    pos: Buf,     // [b] i32
    win_idx: Buf, // [b][w] i32
    blk_idx: Buf, // [b][b] i32
    rl: Buf,      // [b][ne] router logits
    sel: Buf,     // [b][k] i32
    wts: Buf,     // [b][k] f32
    eg: Buf,      // [b][inter] (gate; shared: all rows)
    eu: Buf,      // [b][inter]
    eh: Buf,      // [b][inter]
    ey: Buf,      // [d]
    logits: Buf,  // [b][vocab]
    emb: Buf,     // [rank]
    bias: Buf,    // [vocab]
    best: Buf,    // i32
    mh: Buf,      // [3][d]
    mhmix: Buf,   // [3][24]: 1/hc each
    mx: Buf,      // [d]
    kvo: Buf,     // [hd]
}

pub struct DsparkGpu {
    layers: Vec<DLayer>,
    main_proj: DMat,
    main_norm: Buf,
    norm: Buf,
    markov_embed: Mat,
    markov_head: DMat,
    conf: Vec<f32>,
    b: Bufs,
    pub block: usize,
    pub noise: u32,
    pub targets: Vec<usize>,
    top_k: usize,
    ne: usize,
    /// device memory the stack took (weights and scratch)
    pub bytes: usize,
}

fn up(g: &Gpu, w: &Mat) -> Result<DMat> {
    let scale = if w.scale_bytes().is_empty() { None } else { Some(g.upload(w.scale_bytes())?) };
    Ok(DMat { buf: g.upload(w.bytes())?, scale, ty: w.ty, rows: w.rows, cols: w.cols, act_fp8: w.act_fp8, q8r: false, q6: None })
}

fn f32mat(g: &Gpu, v: &[f32], rows: usize) -> Result<DMat> {
    Ok(DMat { buf: g.upload(v)?, scale: None, ty: GType::F32, rows, cols: v.len() / rows, act_fp8: false, q8r: false, q6: None })
}

fn bf16mat(g: &Gpu, v: &[f32], rows: usize) -> Result<DMat> {
    let h: Vec<half::bf16> = v.iter().map(|&x| half::bf16::from_f32(x)).collect();
    Ok(DMat { buf: g.upload(&h)?, scale: None, ty: GType::BF16, rows, cols: v.len() / rows, act_fp8: false, q8r: false, q6: None })
}

impl DsparkGpu {
    /// Upload the draft stack from the checkpoint directory holding config.json, the index and the mtp.* shards.
    pub fn load(gm: &GpuModel, dir: &Path) -> Result<DsparkGpu> {
        let (m, g): (&Model, &Gpu) = (gm.m, &gm.g);
        let c = &m.cfg;
        g.bind();
        let (free0, _) = g.mem_info()?;
        let st = SafeTensors::open_prefix(dir, "mtp.")?;
        let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).context("mtp config.json")?)?;
        let t = &cfg["text_config"];
        let nl = t["num_nextn_predict_layers"].as_u64().unwrap_or(3) as usize;
        let ne = t["dspark_n_routed_experts"].as_u64().unwrap_or(128) as usize;
        let block = t["dspark_block_size"].as_u64().unwrap_or(5) as usize;
        let top_k = t["dspark_num_experts_per_tok"].as_u64().unwrap_or(3) as usize;
        anyhow::ensure!(block + 1 <= crate::handoff::MAXT && top_k <= 8 && c.hc == 4, "unexpected DSpark config (block {block}, top-k {top_k})");
        let mat = |n: &str| crate::model::hf_mat_pub(&st, n);
        let v = |n: &str| -> Result<Vec<f32>> { Ok(crate::model::mat_f32_pub(&mat(n)?)) };
        let (d, hd, w) = (c.dim, c.head_dim, c.window);
        let mut layers = vec![];
        for l in 0..nl {
            let p = |s: &str| format!("mtp.{l}.{s}");
            let mut o_a = mat(&p("attn.wo_a.weight"))?;
            o_a.act_fp8 = false; // the reference fake-quantizes o per group itself
            let ex = |e: usize, wn: &str| -> Result<DMat> { up(g, &mat(&p(&format!("ffn.experts.{e}.{wn}.weight")))?) };
            layers.push(DLayer {
                attn_norm: f32buf(g, &v(&p("attn_norm.weight"))?)?,
                ffn_norm: f32buf(g, &v(&p("ffn_norm.weight"))?)?,
                q_a: up(g, &mat(&p("attn.wq_a.weight"))?)?,
                q_a_norm: f32buf(g, &v(&p("attn.q_norm.weight"))?)?,
                q_b: up(g, &mat(&p("attn.wq_b.weight"))?)?,
                kv: up(g, &mat(&p("attn.wkv.weight"))?)?,
                kv_norm: f32buf(g, &v(&p("attn.kv_norm.weight"))?)?,
                o_a: up(g, &o_a)?,
                o_b: up(g, &mat(&p("attn.wo_b.weight"))?)?,
                sinks: f32buf(g, &v(&p("attn.attn_sink"))?)?,
                hc_attn: (f32mat(g, &v(&p("hc_attn_fn"))?, 24)?, f32buf(g, &v(&p("hc_attn_scale"))?)?, f32buf(g, &v(&p("hc_attn_base"))?)?),
                hc_ffn: (f32mat(g, &v(&p("hc_ffn_fn"))?, 24)?, f32buf(g, &v(&p("hc_ffn_scale"))?)?, f32buf(g, &v(&p("hc_ffn_base"))?)?),
                gate_w: bf16mat(g, &v(&p("ffn.gate.weight"))?, ne)?,
                gate_bias: f32buf(g, &v(&p("ffn.gate.bias"))?)?,
                experts: (0..ne).map(|e| Ok([ex(e, "w1")?, ex(e, "w3")?, ex(e, "w2")?])).collect::<Result<Vec<_>>>()?,
                sh: [
                    up(g, &mat(&p("ffn.shared_experts.w1.weight"))?)?,
                    up(g, &mat(&p("ffn.shared_experts.w3.weight"))?)?,
                    up(g, &mat(&p("ffn.shared_experts.w2.weight"))?)?,
                ],
                ring: g.alloc_zeroed(w * hd * 2)?,
            });
        }
        let last = nl - 1;
        let markov_embed = mat(&format!("mtp.{last}.markov_head.embed.weight"))?;
        let rank = markov_embed.cols;
        let (b, nh, inter, vocab) = (block, c.n_heads, c.inter, c.vocab);
        let fz = |n: usize| g.alloc_zeroed(n * 4);
        let mut mhmix = vec![0f32; 3 * 24];
        for r in 0..3 {
            for k in 0..4 {
                mhmix[r * 24 + k] = 0.25;
            }
        }
        let bufs = Bufs {
            h: fz(b * 4 * d)?,
            x: fz(b * d)?,
            mix_a: fz(b * 24)?,
            mix_f: fz(b * 24)?,
            qr: fz(b * c.q_lora)?,
            q: fz(b * nh * hd)?,
            kv: fz(b * hd)?,
            blk: g.alloc_zeroed(b * hd * 2)?,
            o: fz(b * nh * hd)?,
            r: fz(b * c.o_groups * c.o_lora)?,
            a: fz(b * d)?,
            sh_out: fz(b * d)?,
            pos: fz(b)?,
            win_idx: fz(b * w)?,
            blk_idx: g.upload(&(0..b * b).map(|i| (i % b) as i32).collect::<Vec<i32>>())?,
            rl: fz(b * ne)?,
            sel: fz(b * top_k)?,
            wts: fz(b * top_k)?,
            eg: fz(b * inter)?,
            eu: fz(b * inter)?,
            eh: fz(b * inter)?,
            ey: fz(d)?,
            logits: fz(b * vocab)?,
            emb: fz(rank)?,
            bias: fz(vocab)?,
            best: fz(1)?,
            mh: fz(3 * d)?,
            mhmix: g.upload(&mhmix)?,
            mx: fz(d)?,
            kvo: fz(hd)?,
        };
        let ds = DsparkGpu {
            layers,
            main_proj: up(g, &mat("mtp.0.main_proj.weight")?)?,
            main_norm: f32buf(g, &v("mtp.0.main_norm.weight")?)?,
            norm: f32buf(g, &v(&format!("mtp.{last}.norm.weight"))?)?,
            markov_head: up(g, &mat(&format!("mtp.{last}.markov_head.head.weight"))?)?,
            conf: v(&format!("mtp.{last}.confidence_head.proj.weight"))?,
            markov_embed,
            b: bufs,
            block,
            noise: t["dspark_noise_token_id"].as_u64().unwrap_or(128799) as u32,
            targets: t["dspark_target_layer_ids"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect()).unwrap_or(vec![37, 38, 39]),
            top_k,
            ne,
            bytes: 0,
        };
        anyhow::ensure!(ds.targets.len() == 3, "DSpark reads 3 target layers, config has {:?}", ds.targets);
        let (free1, _) = g.mem_info()?;
        Ok(DsparkGpu { bytes: free0.saturating_sub(free1), ..ds })
    }

    /// The main model just processed position `pos` (its `step` captured the target layers' residuals): project them
    /// into every draft layer's window ring (the reference `observe`).
    pub fn observe(&mut self, gm: &GpuModel, pos: usize) -> Result<()> {
        let c = &gm.m.cfg;
        let (d, hd, w) = (c.dim, c.head_dim, c.window);
        let hid = gm.hid_buf.as_ref().context("capture_hidden(targets) first")?;
        anyhow::ensure!(gm.hid_layers == self.targets, "captured layers {:?}, DSpark reads {:?}", gm.hid_layers, self.targets);
        let g = &gm.g;
        let b = &self.b;
        // main_hidden: per target layer the mean of the hc streams, bf16 (hc_pre with 1/4 weights rounds)
        g.launch(gm.k["hc_pre"], (3, 1, 1), 256, 0, &[Arg::P(hid.ptr), Arg::P(b.mhmix.ptr), Arg::I(0), Arg::P(b.mh.ptr), Arg::I(d as i32)])?;
        gm.gemv(&self.main_proj, b.mh.ptr, b.mx.ptr)?;
        gm.rmsnorm(b.mx.ptr, &self.main_norm, b.mx.ptr, d, true)?;
        for l in &self.layers {
            gm.gemv(&l.kv, b.mx.ptr, b.kvo.ptr)?;
            g.launch(gm.k["kv_norm_rope_store"], (1, 1, 1), 1024, 0, &[Arg::P(b.kvo.ptr), Arg::P(l.kv_norm.ptr), Arg::I(hd as i32), Arg::F(c.eps), Arg::P(gm.freqs.0.ptr), Arg::I(c.rope_dim as i32), Arg::I(pos as i32), Arg::P(l.ring.ptr + ((pos % w) * hd * 2) as u64)])?;
        }
        Ok(())
    }

    /// Draft `block` tokens after position `pos`, whose next token (sampled by the main model, not yet fed) is
    /// `next`. Returns (drafts, confidence): drafts[i] is the guess for position pos + 2 + i.
    pub fn draft(&mut self, gm: &GpuModel, pos: usize, next: u32) -> Result<(Vec<u32>, Vec<f32>)> {
        let m = gm.m;
        let c = &m.cfg;
        let (d, hc, hd, nh, w, bl) = (c.dim, c.hc, c.head_dim, c.n_heads, c.window, self.block);
        let (inter, vocab, k) = (c.inter, c.vocab, self.top_k);
        let g = &gm.g;
        let b = &self.b;
        let f4 = |n: usize| n * 4;
        // inputs: embeddings (next, then noise), one-hot pre mixes, positions, window rows in use
        let mut h = vec![0f32; bl * hc * d];
        let mut e = vec![0f32; d];
        for i in 0..bl {
            m.embed.dequant(if i == 0 { next } else { self.noise } as usize, &mut e);
            for s in 0..hc {
                h[(i * hc + s) * d..(i * hc + s + 1) * d].copy_from_slice(&e);
            }
        }
        g.htod(b.h.ptr, &h)?;
        let mut pre = vec![0f32; bl * 24];
        for i in 0..bl {
            pre[i * 24] = 1.0;
        }
        g.htod(b.mix_f.ptr, &pre)?;
        g.htod(b.pos.ptr, &(0..bl).map(|i| (pos + 1 + i) as i32).collect::<Vec<i32>>())?;
        let nwin = w.min(pos + 1);
        let widx: Vec<i32> = (0..bl * w).map(|j| if j % w < nwin { (j % w) as i32 } else { -1 }).collect();
        g.htod(b.win_idx.ptr, &widx)?;
        let gsz = nh * hd / c.o_groups;
        for l in &self.layers {
            // attention half, all block rows together
            gm.hc_mixes_rows(&l.hc_attn, b.h.ptr, b.mix_a.ptr, bl)?;
            g.launch(gm.k["hc_pre"], (bl as u32, 1, 1), 256, 0, &[Arg::P(b.h.ptr), Arg::P(b.mix_f.ptr), Arg::I(0), Arg::P(b.x.ptr), Arg::I(d as i32)])?;
            gm.rms_rows(b.x.ptr, &l.attn_norm, b.x.ptr, d, bl, true)?;
            gm.gemv_t(&l.q_a, b.x.ptr, b.qr.ptr, bl)?;
            gm.rms_rows(b.qr.ptr, &l.q_a_norm, b.qr.ptr, c.q_lora, bl, true)?;
            gm.gemv_t(&l.q_b, b.qr.ptr, b.q.ptr, bl)?;
            gm.rope_rows(b.q.ptr, nh, hd, false, b.pos.ptr, bl, false)?;
            gm.gemv_t(&l.kv, b.x.ptr, b.kv.ptr, bl)?;
            for i in 0..bl {
                g.launch(gm.k["kv_norm_rope_store"], (1, 1, 1), 1024, 0, &[Arg::P(b.kv.ptr + f4(i * hd) as u64), Arg::P(l.kv_norm.ptr), Arg::I(hd as i32), Arg::F(c.eps), Arg::P(gm.freqs.0.ptr), Arg::I(c.rope_dim as i32), Arg::I((pos + 1 + i) as i32), Arg::P(b.blk.ptr + (i * hd * 2) as u64)])?;
            }
            // window rows, then every block row (bidirectional), the sink in the softmax; out bf16-rounded
            gm.sparse_attn_small(bl, b.q.ptr, l.ring.ptr, b.win_idx.ptr, w, b.blk.ptr, b.blk_idx.ptr, bl, l.sinks.ptr, b.o.ptr)?;
            gm.rope_rows(b.o.ptr, nh, hd, false, b.pos.ptr, bl, true)?;
            gm.fake_fp8(b.o.ptr, bl * nh * hd)?;
            let rw = c.o_groups * c.o_lora;
            for gi in 0..c.o_groups {
                gm.gemv_rows_ld(&l.o_a, gi * c.o_lora, c.o_lora, b.o.ptr + f4(gi * gsz) as u64, nh * hd, b.r.ptr + f4(gi * c.o_lora) as u64, rw, bl)?;
            }
            g.launch(gm.k["round_bf16"], (((bl * rw) as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(b.r.ptr), Arg::I((bl * rw) as i32)])?;
            gm.gemv_t(&l.o_b, b.r.ptr, b.a.ptr, bl)?;
            g.launch(gm.k["hc_post"], (bl as u32, 1, 1), 256, 0, &[Arg::P(b.h.ptr), Arg::P(b.a.ptr), Arg::P(b.mix_a.ptr), Arg::I(d as i32)])?;
            // ffn half: input from the attention mixes' pre, this half's mixes for its post and the next layer's pre
            g.launch(gm.k["hc_pre"], (bl as u32, 1, 1), 256, 0, &[Arg::P(b.h.ptr), Arg::P(b.mix_a.ptr), Arg::I(0), Arg::P(b.x.ptr), Arg::I(d as i32)])?;
            gm.hc_mixes_rows(&l.hc_ffn, b.h.ptr, b.mix_f.ptr, bl)?;
            gm.rms_rows(b.x.ptr, &l.ffn_norm, b.x.ptr, d, bl, true)?;
            self.moe(gm, l, bl, k, inter)?;
            g.launch(gm.k["hc_post"], (bl as u32, 1, 1), 256, 0, &[Arg::P(b.h.ptr), Arg::P(b.a.ptr), Arg::P(b.mix_f.ptr), Arg::I(d as i32)])?;
        }
        // head: the main model's logits per row, then in order the Markov bias of the previous token and greedy
        g.launch(gm.k["hc_pre"], (bl as u32, 1, 1), 256, 0, &[Arg::P(b.h.ptr), Arg::P(b.mix_f.ptr), Arg::I(0), Arg::P(b.x.ptr), Arg::I(d as i32)])?;
        gm.rms_rows(b.x.ptr, &self.norm, b.x.ptr, d, bl, false)?;
        gm.gemv_t(&gm.head, b.x.ptr, b.logits.ptr, bl)?;
        let xs: Vec<f32> = g.download(b.x.ptr, bl * d)?;
        let rank = self.markov_embed.cols;
        let mut emb = vec![0f32; rank];
        let (mut out, mut conf) = (vec![next], vec![]);
        for i in 0..bl {
            self.markov_embed.dequant(out[i] as usize, &mut emb);
            g.htod(b.emb.ptr, &emb)?;
            gm.gemv(&self.markov_head, b.emb.ptr, b.bias.ptr)?;
            g.launch(gm.k["dsp_argmax_add"], (1, 1, 1), 1024, 0, &[Arg::P(b.logits.ptr + f4(i * vocab) as u64), Arg::P(b.bias.ptr), Arg::I(vocab as i32), Arg::P(b.best.ptr)])?;
            out.push(g.download::<i32>(b.best.ptr, 1)?[0] as u32);
            let x = &xs[i * d..(i + 1) * d];
            conf.push(x.iter().chain(&emb).zip(&self.conf).map(|(a, b)| a * b).sum());
        }
        Ok((out[1..].to_vec(), conf))
    }

    /// x [bl][d] (normed) -> a [bl][d]: routed experts (per row and pick) then the shared expert, bf16
    fn moe(&self, gm: &GpuModel, l: &DLayer, bl: usize, k: usize, inter: usize) -> Result<()> {
        let c = &gm.m.cfg;
        let d = c.dim;
        let (g, b) = (&gm.g, &self.b);
        let f4 = |n: usize| (n * 4) as u64;
        gm.gemv_t(&l.gate_w, b.x.ptr, b.rl.ptr, bl)?;
        g.launch(gm.k["router_topk"], (bl as u32, 1, 1), 32, 0, &[Arg::P(b.rl.ptr), Arg::P(l.gate_bias.ptr), Arg::P(b.sel.ptr), Arg::P(b.wts.ptr), Arg::I(self.ne as i32), Arg::I(k as i32), Arg::F(c.route_scale), Arg::P(0)])?;
        // shared expert for every row (into sh_out) while the picks come back
        gm.gemv_t(&l.sh[0], b.x.ptr, b.eg.ptr, bl)?;
        gm.gemv_t(&l.sh[1], b.x.ptr, b.eu.ptr, bl)?;
        g.launch(gm.k["swiglu"], ((inter as u32).div_ceil(256), bl as u32, 1), 256, 0, &[Arg::P(b.eg.ptr), Arg::P(b.eu.ptr), Arg::I(inter as i32), Arg::P(0), Arg::P(b.eh.ptr), Arg::I(inter as i32), Arg::F(c.swiglu_limit)])?;
        gm.gemv_t(&l.sh[2], b.eh.ptr, b.sh_out.ptr, bl)?;
        let sel: Vec<i32> = g.download(b.sel.ptr, bl * k)?;
        g.memset(b.a.ptr, bl * d * 4)?;
        for t in 0..bl {
            let xt = b.x.ptr + f4(t * d);
            for j in 0..k {
                let e = sel[t * k + j];
                anyhow::ensure!(e >= 0 && (e as usize) < self.ne, "draft router picked expert {e}");
                let ex = &l.experts[e as usize];
                gm.gemv_t(&ex[0], xt, b.eg.ptr, 1)?;
                gm.gemv_t(&ex[1], xt, b.eu.ptr, 1)?;
                g.launch(gm.k["swiglu"], ((inter as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(b.eg.ptr), Arg::P(b.eu.ptr), Arg::I(inter as i32), Arg::P(b.wts.ptr + f4(t * k + j)), Arg::P(b.eh.ptr), Arg::I(inter as i32), Arg::F(c.swiglu_limit)])?;
                gm.gemv_t(&ex[2], b.eh.ptr, b.ey.ptr, 1)?;
                g.launch(gm.k["add_inplace"], ((d as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(b.a.ptr + f4(t * d)), Arg::P(b.ey.ptr), Arg::I(d as i32)])?;
            }
        }
        g.launch(gm.k["add_inplace"], (((bl * d) as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(b.a.ptr), Arg::P(b.sh_out.ptr), Arg::I((bl * d) as i32)])?;
        g.launch(gm.k["round_bf16"], (((bl * d) as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(b.a.ptr), Arg::I((bl * d) as i32)])
    }
}
