//! Speculative decoding drafts that cost nothing to make: continuations looked up in the context itself.

/// Up to `k` tokens that followed the most recent earlier occurrence of the context's last 3 (else 2) tokens.
pub fn lookup(ctx: &[u32], k: usize) -> Vec<u32> {
    for n in [3usize, 2] {
        if ctx.len() <= n {
            continue;
        }
        let tail = &ctx[ctx.len() - n..];
        // newest match first; the match must leave at least one token after it
        for s in (0..ctx.len() - n).rev() {
            if &ctx[s..s + n] == tail {
                let from = s + n;
                let to = (from + k).min(ctx.len());
                if to > from {
                    return ctx[from..to].to_vec();
                }
            }
        }
    }
    vec![]
}


/// Live control of speculative decoding, the same on any hardware: it measures what a plain step, a draft and a
/// verify pass of T tokens cost here, and how often each draft position hits, and before every token picks the draft
/// length with the most tokens per millisecond (0 = decode plainly). While plain decoding wins it drafts again (a
/// probe, 2 drafts) after a gap that doubles while probes do not pay (16 .. 256 tokens), and early when the main
/// model has grown as confident as it was when drafting last paid (text turning into code).
#[derive(Clone, Debug)]
pub struct SpecControl {
    /// ms per plain step, per draft, per verify pass of T tokens (index T, 1..=max+1)
    pub step_ms: f64,
    pub draft_ms: f64,
    pub verify_ms: Vec<f64>,
    /// hit rate of draft position i given the positions before it hit
    pub hit: Vec<f64>,
    pub max: usize,
    since_probe: usize,
    pub probe_gap: usize,
    probing: bool,
    /// the main model's top probability (EMA over tokens) and its level when drafting last paid
    pub conf: f64,
    pub conf_paid: f64,
}

pub const PROBE_MIN: usize = 16;
pub const PROBE_MAX: usize = 256;

impl SpecControl {
    pub fn new(max: usize) -> SpecControl {
        // priors: a verify pass of T tokens ~ 1 + 0.5 (T - 1) steps, drafts at 0.4 steps, positions hit 80 %
        SpecControl {
            step_ms: 25.0,
            draft_ms: 10.0,
            verify_ms: (0..=max + 1).map(|t| 25.0 * (1.0 + 0.5 * (t.max(1) - 1) as f64)).collect(),
            hit: vec![0.8; max],
            max,
            since_probe: 0,
            probe_gap: PROBE_MIN,
            probing: false,
            conf: 0.0,
            conf_paid: 1.0,
        }
    }

    fn ema(x: &mut f64, v: f64, a: f64) {
        *x = (1.0 - a) * *x + a * v;
    }

    /// expected tokens per ms when drafting `l` (0 = plain step)
    pub fn rate(&self, l: usize) -> f64 {
        if l == 0 {
            return 1.0 / self.step_ms;
        }
        let (mut reach, mut exp) = (1.0, 1.0);
        for i in 0..l {
            reach *= self.hit[i];
            exp += reach;
        }
        exp / (self.draft_ms + self.verify_ms[l + 1])
    }

    /// draft length for the next token (0 = plain), at most `room`
    pub fn choose(&mut self, room: usize) -> usize {
        let room = room.min(self.max);
        let rates: Vec<f64> = (0..=room).map(|l| self.rate(l)).collect();
        let top = rates.iter().cloned().fold(0.0, f64::max);
        // within 2 % the longer draft: its estimate rests on more evidence and the verify cost is flat per token
        let best = (0..=room).rev().find(|&l| rates[l] >= 0.98 * top).unwrap_or(0);
        self.probing = false;
        if best > 0 || room == 0 {
            self.since_probe = 0;
            return best;
        }
        self.since_probe += 1;
        // early, but still within the back-off: at most every quarter gap
        let confident = self.since_probe >= (self.probe_gap / 4).max(4) && self.conf >= self.conf_paid;
        if self.since_probe >= self.probe_gap || confident {
            self.since_probe = 0;
            self.probing = true;
            return room.min(2);
        }
        0
    }

    pub fn note_step(&mut self, ms: f64) {
        Self::ema(&mut self.step_ms, ms, 0.1);
    }

    /// the main model's top probability for the token just decided (any path)
    pub fn note_conf(&mut self, pmax: f64) {
        Self::ema(&mut self.conf, pmax, 0.2);
    }

    /// a pass drafted `l` (taking `draft_ms`), verified them in `verify_ms`, accepted `j`
    pub fn note_pass(&mut self, l: usize, j: usize, draft_ms: f64, verify_ms: f64) {
        Self::ema(&mut self.draft_ms, draft_ms, 0.2);
        Self::ema(&mut self.verify_ms[l + 1], verify_ms, 0.2);
        for i in 0..l.min(j + 1) {
            Self::ema(&mut self.hit[i], if i < j { 1.0 } else { 0.0 }, 0.15);
        }
        let paid = (1 + j) as f64 / (draft_ms + verify_ms) > 1.0 / self.step_ms;
        if paid {
            Self::ema(&mut self.conf_paid, self.conf, 0.3);
            self.probe_gap = PROBE_MIN;
        } else if self.probing {
            self.probe_gap = (self.probe_gap * 2).min(PROBE_MAX);
        }
        self.probing = false;
    }
}

/// The top probability of a logit vector (softmax max), for SpecControl::note_conf
/// (every token: only logits within 16 of the max are exponentiated, the rest weigh < 1e-7 each)
pub fn top_prob(logits: &[f32]) -> f64 {
    let mx = logits.iter().cloned().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, f32::max);
    if !mx.is_finite() {
        return 0.0;
    }
    let th = mx - 16.0;
    let sum: f32 = logits.iter().filter(|&&v| v > th).map(|&v| (v - mx).exp()).sum();
    1.0 / (sum as f64).max(1.0)
}

#[cfg(test)]
mod tests {
    #[test]
    fn finds_the_newest_continuation() {
        let c = [1, 2, 3, 9, 9, 1, 2, 3, 7, 8, 5, 1, 2, 3];
        assert_eq!(super::lookup(&c, 2), vec![7, 8]);
        assert_eq!(super::lookup(&[4, 5, 6], 3), Vec::<u32>::new());
    }

    #[test]
    fn control_drafts_when_it_pays_and_stops_when_not() {
        let mut c = super::SpecControl::new(5);
        c.step_ms = 25.0;
        c.draft_ms = 10.0;
        c.verify_ms = vec![25.0, 25.0, 58.0, 70.0, 82.0, 88.0, 95.0];
        c.hit = vec![0.95, 0.93, 0.88, 0.87, 0.85];
        assert_eq!(c.choose(5), 5);
        c.hit = vec![0.65, 0.5, 0.45, 0.5, 0.4];
        assert_eq!(c.choose(5), 0);
        // probes now and then, less often while they do not pay
        let first = (1..=20).find(|_| c.choose(5) > 0).unwrap();
        assert!(first <= 16);
        c.note_pass(2, 0, 10.0, 58.0);
        assert_eq!(c.probe_gap, 32);
        // the main model as confident as when drafting paid: probe early
        c.conf_paid = 0.8;
        c.note_conf(1.0);
        c.note_conf(1.0);
        c.note_conf(1.0);
        c.note_conf(1.0);
        c.note_conf(1.0);
        c.note_conf(1.0);
        c.note_conf(1.0);
        c.note_conf(1.0);
        assert!((1..=8).any(|_| c.choose(5) > 0));
        assert!((super::top_prob(&[0.0, 0.0]) - 0.5).abs() < 1e-9);
    }
}
