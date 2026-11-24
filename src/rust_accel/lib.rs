#![allow(non_camel_case_types)]

use std::slice;

const EMPTY_TARGET: u32 = u32::MAX;
const MAX_STATS_WALK: usize = 24;
const INLINE_CAP: usize = 4;

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct NodeStats {
    total_cnt: u32,
    best_tok: u32,
    best_cnt: u32,
    second_tok: u32,
    second_cnt: u32,
}

impl NodeStats {
    #[inline]
    fn record(&mut self, tok: u32) {
        self.total_cnt = self.total_cnt.saturating_add(1);
        if self.best_cnt == 0 {
            self.best_tok = tok;
            self.best_cnt = 1;
            return;
        }
        if tok == self.best_tok {
            self.best_cnt = self.best_cnt.saturating_add(1);
            return;
        }
        if self.second_cnt == 0 {
            self.second_tok = tok;
            self.second_cnt = 1;
            return;
        }
        if tok == self.second_tok {
            self.second_cnt = self.second_cnt.saturating_add(1);
            if self.second_cnt > self.best_cnt {
                std::mem::swap(&mut self.best_tok, &mut self.second_tok);
                std::mem::swap(&mut self.best_cnt, &mut self.second_cnt);
            }
            return;
        }
        // Conservative heavy-hitter replacement: decrement second_cnt down to 0 before
        // evicting so best_cnt/second_cnt never exceed true continuation counts.
        self.second_cnt -= 1;
        if self.second_cnt == 0 {
            self.second_tok = tok;
            self.second_cnt = 1;
        }
    }
}

#[inline]
fn hash_tok(tok: u32) -> usize {
    let mut x = tok.wrapping_mul(0x9e37_79b9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85eb_ca6b);
    (x ^ (x >> 13)) as usize
}

#[derive(Clone)]
struct SmallEdgeHash {
    slots: Vec<(u32, u32)>,
    len: usize,
}

impl SmallEdgeHash {
    fn with_capacity(cap: usize) -> Self {
        let cap = cap.max(16).next_power_of_two();
        Self {
            slots: vec![(0, EMPTY_TARGET); cap],
            len: 0,
        }
    }

    #[inline]
    fn get(&self, tok: u32) -> i32 {
        let mask = self.slots.len() - 1;
        let mut idx = hash_tok(tok) & mask;
        loop {
            let (k, v) = self.slots[idx];
            if v == EMPTY_TARGET {
                return -1;
            }
            if k == tok {
                return v as i32;
            }
            idx = (idx + 1) & mask;
        }
    }

    fn rehash(&mut self, new_cap: usize) {
        let mut next = vec![(0u32, EMPTY_TARGET); new_cap];
        let mask = new_cap - 1;
        for &(k, v) in &self.slots {
            if v == EMPTY_TARGET {
                continue;
            }
            let mut idx = hash_tok(k) & mask;
            while next[idx].1 != EMPTY_TARGET {
                idx = (idx + 1) & mask;
            }
            next[idx] = (k, v);
        }
        self.slots = next;
    }

    #[inline]
    fn set(&mut self, tok: u32, target: u32) -> i32 {
        if (self.len + 1) * 10 > self.slots.len() * 7 {
            self.rehash(self.slots.len() * 2);
        }
        let mask = self.slots.len() - 1;
        let mut idx = hash_tok(tok) & mask;
        loop {
            let (k, v) = self.slots[idx];
            if v == EMPTY_TARGET {
                self.slots[idx] = (tok, target);
                self.len += 1;
                return -1;
            }
            if k == tok {
                self.slots[idx].1 = target;
                return v as i32;
            }
            idx = (idx + 1) & mask;
        }
    }

    fn remove(&mut self, tok: u32) {
        let mask = self.slots.len() - 1;
        let mut idx = hash_tok(tok) & mask;
        loop {
            let (k, v) = self.slots[idx];
            if v == EMPTY_TARGET {
                return;
            }
            if k == tok {
                break;
            }
            idx = (idx + 1) & mask;
        }
        // Backward-shift deletion preserves linear-probe invariants without tombstones.
        let mut empty = idx;
        let mut j = (idx + 1) & mask;
        while self.slots[j].1 != EMPTY_TARGET {
            let natural = hash_tok(self.slots[j].0) & mask;
            if (j.wrapping_sub(natural) & mask) >= (j.wrapping_sub(empty) & mask) {
                self.slots[empty] = self.slots[j];
                empty = j;
            }
            j = (j + 1) & mask;
        }
        self.slots[empty] = (0, EMPTY_TARGET);
        self.len -= 1;
    }
}

#[derive(Clone, Copy)]
struct State {
    len: u32,
    link: i32,
    first_endpos: u32,
    stats: NodeStats,
    inline_edges: [(u32, u32); INLINE_CAP],
    n_inline: u8,
    edge_map: i32,
}

impl State {
    fn new(len: u32, link: i32, first_endpos: u32) -> Self {
        Self {
            len,
            link,
            first_endpos,
            stats: NodeStats::default(),
            inline_edges: [(0, 0); INLINE_CAP],
            n_inline: 0,
            edge_map: -1,
        }
    }
}

#[derive(Clone, Copy)]
enum UndoKind {
    SetLink = 0,
    RestoreStats = 1,
    RestoreEdge = 2,
    DemoteEdgeMap = 3,
}

// Compact 12-byte undo entry; stats payloads live in a side vector.
#[derive(Clone, Copy)]
struct UndoOp {
    kind_state: u32,
    arg0: u32,
    arg1: i32,
}

impl UndoOp {
    #[inline]
    fn pack(kind: UndoKind, state: u32, arg0: u32, arg1: i32) -> Self {
        Self {
            kind_state: ((kind as u32) << 30) | (state & 0x3FFF_FFFF),
            arg0,
            arg1,
        }
    }

    #[inline]
    fn kind(self) -> UndoKind {
        match self.kind_state >> 30 {
            0 => UndoKind::SetLink,
            1 => UndoKind::RestoreStats,
            2 => UndoKind::RestoreEdge,
            _ => UndoKind::DemoteEdgeMap,
        }
    }

    #[inline]
    fn state(self) -> usize {
        (self.kind_state & 0x3FFF_FFFF) as usize
    }
}

#[derive(Clone, Copy)]
struct StepMeta {
    prev_num_states: u32,
    prev_num_edge_maps: u32,
    prev_stats_undo: u32,
    prev_last: u32,
    prev_cur_state: u32,
    prev_match_len: u32,
    undo_start: u32,
}

pub struct AffSam {
    tokens: Vec<u32>,
    states: Vec<State>,
    edge_maps: Vec<SmallEdgeHash>,
    last: u32,
    cur_state: u32,
    match_len: u32,
    step_meta: Vec<StepMeta>,
    undo_ops: Vec<UndoOp>,
    stats_undo: Vec<NodeStats>,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct AffSamProposal {
    pub len: usize,
    pub match_len: usize,
    pub score: f32,
    pub expect: f32,
}

impl AffSam {
    pub fn new() -> Self {
        let mut sam = Self {
            tokens: Vec::new(),
            states: Vec::with_capacity(256),
            edge_maps: Vec::new(),
            last: 0,
            cur_state: 0,
            match_len: 0,
            step_meta: Vec::new(),
            undo_ops: Vec::new(),
            stats_undo: Vec::new(),
        };
        sam.states.push(State::new(0, -1, 0));
        sam
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.states.clear();
        self.states.push(State::new(0, -1, 0));
        self.edge_maps.clear();
        self.last = 0;
        self.cur_state = 0;
        self.match_len = 0;
        self.step_meta.clear();
        self.undo_ops.clear();
        self.stats_undo.clear();
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    #[inline]
    pub fn match_len(&self) -> usize {
        self.match_len as usize
    }

    #[inline]
    fn get_trans(&self, state: usize, tok: u32) -> i32 {
        let st = &self.states[state];
        if st.edge_map >= 0 {
            return self.edge_maps[st.edge_map as usize].get(tok);
        }
        let n = st.n_inline as usize;
        for i in 0..n {
            if st.inline_edges[i].0 == tok {
                return st.inline_edges[i].1 as i32;
            }
        }
        -1
    }

    #[inline]
    fn set_trans_logged(&mut self, state: usize, tok: u32, target: u32, prev_num_states: u32) {
        let log_mutation = (state as u32) < prev_num_states;
        let map_idx = self.states[state].edge_map;
        if map_idx >= 0 {
            let old = self.edge_maps[map_idx as usize].set(tok, target);
            if log_mutation {
                self.undo_ops
                    .push(UndoOp::pack(UndoKind::RestoreEdge, state as u32, tok, old));
            }
            return;
        }
        let n = self.states[state].n_inline as usize;
        for i in 0..n {
            if self.states[state].inline_edges[i].0 == tok {
                let old = self.states[state].inline_edges[i].1 as i32;
                self.states[state].inline_edges[i].1 = target;
                if log_mutation {
                    self.undo_ops
                        .push(UndoOp::pack(UndoKind::RestoreEdge, state as u32, tok, old));
                }
                return;
            }
        }
        if n < INLINE_CAP {
            self.states[state].inline_edges[n] = (tok, target);
            self.states[state].n_inline = (n + 1) as u8;
            if log_mutation {
                self.undo_ops
                    .push(UndoOp::pack(UndoKind::RestoreEdge, state as u32, tok, -1));
            }
            return;
        }
        // Promote inline edges to an open-addressed table while keeping inline_edges intact
        // so DemoteEdgeMap restores all 4 inline edges in O(1).
        let mut h = SmallEdgeHash::with_capacity(16);
        for i in 0..INLINE_CAP {
            let (ek, ev) = self.states[state].inline_edges[i];
            h.set(ek, ev);
        }
        h.set(tok, target);
        let new_map_id = self.edge_maps.len() as i32;
        self.edge_maps.push(h);
        self.states[state].edge_map = new_map_id;
        if log_mutation {
            self.undo_ops
                .push(UndoOp::pack(UndoKind::DemoteEdgeMap, state as u32, 0, 0));
        }
    }

    fn undo_remove_edge(&mut self, state: usize, tok: u32) {
        let map_idx = self.states[state].edge_map;
        if map_idx >= 0 {
            self.edge_maps[map_idx as usize].remove(tok);
            return;
        }
        let n = self.states[state].n_inline as usize;
        for i in 0..n {
            if self.states[state].inline_edges[i].0 == tok {
                if i + 1 < n {
                    self.states[state].inline_edges[i] = self.states[state].inline_edges[n - 1];
                }
                self.states[state].n_inline = (n - 1) as u8;
                return;
            }
        }
    }

    fn undo_restore_edge(&mut self, state: usize, tok: u32, old_target: u32) {
        let map_idx = self.states[state].edge_map;
        if map_idx >= 0 {
            self.edge_maps[map_idx as usize].set(tok, old_target);
            return;
        }
        let n = self.states[state].n_inline as usize;
        for i in 0..n {
            if self.states[state].inline_edges[i].0 == tok {
                self.states[state].inline_edges[i].1 = old_target;
                return;
            }
        }
    }

    fn clone_state(&mut self, q: usize, new_len: u32) -> u32 {
        let mut cloned = self.states[q];
        cloned.len = new_len;
        if cloned.edge_map >= 0 {
            let dup = self.edge_maps[cloned.edge_map as usize].clone();
            cloned.edge_map = self.edge_maps.len() as i32;
            self.edge_maps.push(dup);
        }
        let idx = self.states.len() as u32;
        self.states.push(cloned);
        idx
    }

    pub fn push(&mut self, tok: u32) {
        let pos = self.tokens.len();
        let prev_num_states = self.states.len() as u32;
        let prev_num_edge_maps = self.edge_maps.len() as u32;
        let prev_stats_undo = self.stats_undo.len() as u32;
        let undo_start = self.undo_ops.len() as u32;

        self.step_meta.push(StepMeta {
            prev_num_states,
            prev_num_edge_maps,
            prev_stats_undo,
            prev_last: self.last,
            prev_cur_state: self.cur_state,
            prev_match_len: self.match_len,
            undo_start,
        });

        // Phase 1: record empirical continuation `tok` along the suffix-link chain of `last`
        // (capped to MAX_STATS_WALK so adversarial periodic runs remain strictly O(1)).
        let mut sp = self.last as i32;
        let mut walk_steps = 0usize;
        while sp > 0 && walk_steps < MAX_STATS_WALK {
            let su = sp as usize;
            let s_idx = self.stats_undo.len() as u32;
            self.stats_undo.push(self.states[su].stats);
            self.undo_ops
                .push(UndoOp::pack(UndoKind::RestoreStats, sp as u32, s_idx, 0));
            self.states[su].stats.record(tok);
            sp = self.states[su].link;
            walk_steps += 1;
        }

        // Phase 2: standard online Suffix Automaton extension.
        let cur = self.states.len() as u32;
        let cur_len = self.states[self.last as usize].len + 1;
        let endpos = (pos + 1) as u32;
        self.states.push(State::new(cur_len, 0, endpos));

        let mut p = self.last as i32;
        while p != -1 && self.get_trans(p as usize, tok) < 0 {
            self.set_trans_logged(p as usize, tok, cur, prev_num_states);
            p = self.states[p as usize].link;
        }

        if p == -1 {
            self.states[cur as usize].link = 0;
            self.cur_state = 0;
            self.match_len = 0;
        } else {
            let q = self.get_trans(p as usize, tok) as u32;
            if self.states[p as usize].len + 1 == self.states[q as usize].len {
                self.states[cur as usize].link = q as i32;
                self.cur_state = q;
                self.match_len = self.states[q as usize].len;
            } else {
                let clone_len = self.states[p as usize].len + 1;
                let clone = self.clone_state(q as usize, clone_len);
                while p != -1 && self.get_trans(p as usize, tok) == q as i32 {
                    self.set_trans_logged(p as usize, tok, clone, prev_num_states);
                    p = self.states[p as usize].link;
                }
                if q < prev_num_states {
                    self.undo_ops.push(UndoOp::pack(
                        UndoKind::SetLink,
                        q,
                        0,
                        self.states[q as usize].link,
                    ));
                }
                self.states[q as usize].link = clone as i32;
                self.states[cur as usize].link = clone as i32;
                self.cur_state = clone;
                self.match_len = clone_len;
            }
        }

        self.last = cur;
        self.tokens.push(tok);
    }

    pub fn push_slice(&mut self, slice: &[u32]) {
        self.tokens.reserve(slice.len());
        self.step_meta.reserve(slice.len());
        self.states.reserve(slice.len().saturating_mul(2));
        self.undo_ops.reserve(slice.len().saturating_mul(8));
        self.stats_undo.reserve(slice.len().saturating_mul(8));
        for &tok in slice {
            self.push(tok);
        }
    }

    pub fn rollback_to(&mut self, target_len: usize) -> bool {
        if target_len > self.tokens.len() {
            return false;
        }
        if target_len == self.tokens.len() {
            return true;
        }
        if target_len == 0 {
            self.clear();
            return true;
        }
        let target_meta = self.step_meta[target_len];
        let mut op_idx = self.undo_ops.len();
        let stop_idx = target_meta.undo_start as usize;
        while op_idx > stop_idx {
            op_idx -= 1;
            let op = self.undo_ops[op_idx];
            let st = op.state();
            match op.kind() {
                UndoKind::SetLink => {
                    self.states[st].link = op.arg1;
                }
                UndoKind::RestoreStats => {
                    self.states[st].stats = self.stats_undo[op.arg0 as usize];
                }
                UndoKind::RestoreEdge => {
                    if op.arg1 < 0 {
                        self.undo_remove_edge(st, op.arg0);
                    } else {
                        self.undo_restore_edge(st, op.arg0, op.arg1 as u32);
                    }
                }
                UndoKind::DemoteEdgeMap => {
                    self.states[st].edge_map = -1;
                }
            }
        }
        self.undo_ops.truncate(stop_idx);
        self.stats_undo.truncate(target_meta.prev_stats_undo as usize);
        self.edge_maps
            .truncate(target_meta.prev_num_edge_maps as usize);
        self.states.truncate(target_meta.prev_num_states as usize);
        self.step_meta.truncate(target_len);
        self.tokens.truncate(target_len);
        self.last = target_meta.prev_last;
        self.cur_state = target_meta.prev_cur_state;
        self.match_len = target_meta.prev_match_len;
        true
    }

    #[inline]
    fn find_ngram_state(&self, ngram: &[u32]) -> Option<usize> {
        if ngram.is_empty() || ngram.len() > self.tokens.len() {
            return None;
        }
        let mut u: usize = 0;
        for &tok in ngram {
            let nxt = self.get_trans(u, tok);
            if nxt < 0 {
                return None;
            }
            u = nxt as usize;
        }
        Some(u)
    }

    #[inline]
    pub fn contains_ngram(&self, ngram: &[u32]) -> bool {
        self.find_ngram_state(ngram).is_some()
    }

    #[inline]
    pub fn first_endpos(&self, ngram: &[u32]) -> usize {
        match self.find_ngram_state(ngram) {
            Some(u) => self.states[u].first_endpos as usize,
            None => 0,
        }
    }

    pub fn propose_impl(
        &self,
        ctx: &[u32],
        max_draft: usize,
        min_match: usize,
        min_score: f32,
        out_toks: &mut [u32],
    ) -> AffSamProposal {
        if ctx.is_empty() || max_draft == 0 || min_match == 0 || out_toks.len() < max_draft {
            return AffSamProposal::default();
        }
        if !min_score.is_finite() {
            return AffSamProposal::default();
        }
        let min_score = min_score.max(0.0);

        let (start_state, matched_len) = if self.tokens.ends_with(ctx) {
            if (self.match_len as usize) < min_match
                || self.cur_state == 0
                || ctx.len() < min_match
            {
                return AffSamProposal::default();
            }
            if ctx.len() >= self.match_len as usize {
                (self.cur_state as usize, self.match_len as usize)
            } else {
                let mut u = self.cur_state as usize;
                while u != 0 {
                    let lk = self.states[u].link;
                    if lk <= 0 || (self.states[lk as usize].len as usize) < ctx.len() {
                        break;
                    }
                    u = lk as usize;
                }
                (u, ctx.len().min(self.states[u].len as usize))
            }
        } else {
            let mut u: usize = 0;
            let mut mlen: usize = 0;
            for &tok in ctx {
                while u != 0 && self.get_trans(u, tok) < 0 {
                    u = self.states[u].link as usize;
                    mlen = self.states[u].len as usize;
                }
                let nxt = self.get_trans(u, tok);
                if nxt >= 0 {
                    u = nxt as usize;
                    mlen += 1;
                } else {
                    u = 0;
                    mlen = 0;
                }
            }
            while u != 0 && self.states[u].stats.total_cnt == 0 {
                let lk = self.states[u].link;
                if lk <= 0 {
                    u = 0;
                    mlen = 0;
                    break;
                }
                u = lk as usize;
                mlen = self.states[u].len as usize;
            }
            if u == 0 || mlen < min_match {
                return AffSamProposal::default();
            }
            (u, mlen)
        };

        if self.states[start_state].stats.total_cnt == 0 {
            return AffSamProposal::default();
        }

        let mut cur = start_state;
        let mut d_cum: f32 = 1.0;
        let mut last_d: f32 = 0.0;
        let mut expect: f32 = 0.0;
        let mut kept: usize = 0;

        for slot in out_toks.iter_mut().take(max_draft) {
            let st = &self.states[cur];
            if st.stats.total_cnt == 0 || st.stats.best_cnt == 0 {
                break;
            }
            let tok = st.stats.best_tok;
            let p = (st.stats.best_cnt as f32) / (st.stats.total_cnt as f32);
            d_cum *= p;
            if d_cum < min_score {
                break;
            }
            *slot = tok;
            kept += 1;
            last_d = d_cum;
            expect += d_cum;

            let nxt = self.get_trans(cur, tok);
            if nxt < 0 {
                break;
            }
            cur = nxt as usize;
        }

        if kept == 0 {
            return AffSamProposal::default();
        }

        AffSamProposal {
            len: kept,
            match_len: matched_len,
            score: last_d,
            expect,
        }
    }
}

impl Default for AffSam {
    fn default() -> Self {
        Self::new()
    }
}

#[no_mangle]
pub extern "C" fn aff_sam_new() -> *mut AffSam {
    Box::into_raw(Box::new(AffSam::new()))
}

#[no_mangle]
pub unsafe extern "C" fn aff_sam_free(sam: *mut AffSam) {
    if !sam.is_null() {
        drop(Box::from_raw(sam));
    }
}

#[no_mangle]
pub unsafe extern "C" fn aff_sam_clear(sam: *mut AffSam) {
    if !sam.is_null() {
        (*sam).clear();
    }
}

#[no_mangle]
pub unsafe extern "C" fn aff_sam_len(sam: *const AffSam) -> usize {
    if sam.is_null() {
        0
    } else {
        (*sam).len()
    }
}

#[no_mangle]
pub unsafe extern "C" fn aff_sam_push(sam: *mut AffSam, tok: u32) {
    if !sam.is_null() {
        (*sam).push(tok);
    }
}

#[no_mangle]
pub unsafe extern "C" fn aff_sam_push_slice(sam: *mut AffSam, ptr: *const u32, len: usize) {
    if !sam.is_null() && !ptr.is_null() && len > 0 {
        let s = slice::from_raw_parts(ptr, len);
        (*sam).push_slice(s);
    }
}

#[no_mangle]
pub unsafe extern "C" fn aff_sam_rollback_to(sam: *mut AffSam, target_len: usize) -> bool {
    if sam.is_null() {
        false
    } else {
        (*sam).rollback_to(target_len)
    }
}

#[no_mangle]
pub unsafe extern "C" fn aff_sam_match_len(sam: *const AffSam) -> usize {
    if sam.is_null() {
        0
    } else {
        (*sam).match_len()
    }
}

#[no_mangle]
pub unsafe extern "C" fn aff_sam_propose_ex(
    sam: *const AffSam,
    ctx_ptr: *const u32,
    ctx_len: usize,
    max_draft: usize,
    min_match: usize,
    min_score: f32,
    out_toks: *mut u32,
    out_len: *mut usize,
    out_score: *mut f32,
    out_expect: *mut f32,
    out_match: *mut usize,
) -> bool {
    if !out_len.is_null() {
        *out_len = 0;
    }
    if !out_score.is_null() {
        *out_score = 0.0;
    }
    if !out_expect.is_null() {
        *out_expect = 0.0;
    }
    if !out_match.is_null() {
        *out_match = 0;
    }
    if sam.is_null() || ctx_ptr.is_null() || ctx_len == 0 || max_draft == 0 || out_toks.is_null() {
        return false;
    }
    let ctx = slice::from_raw_parts(ctx_ptr, ctx_len);
    let out = slice::from_raw_parts_mut(out_toks, max_draft);
    let prop = (*sam).propose_impl(ctx, max_draft, min_match, min_score, out);
    if prop.len == 0 {
        return false;
    }
    if !out_len.is_null() {
        *out_len = prop.len;
    }
    if !out_score.is_null() {
        *out_score = prop.score;
    }
    if !out_expect.is_null() {
        *out_expect = prop.expect;
    }
    if !out_match.is_null() {
        *out_match = prop.match_len;
    }
    true
}

#[no_mangle]
pub unsafe extern "C" fn aff_sam_propose(
    sam: *const AffSam,
    ctx_ptr: *const u32,
    ctx_len: usize,
    max_draft: usize,
    min_match: usize,
    min_score: f32,
    out_toks: *mut u32,
    out_len: *mut usize,
    out_score: *mut f32,
    out_match: *mut usize,
) -> bool {
    aff_sam_propose_ex(
        sam,
        ctx_ptr,
        ctx_len,
        max_draft,
        min_match,
        min_score,
        out_toks,
        out_len,
        out_score,
        std::ptr::null_mut(),
        out_match,
    )
}

#[no_mangle]
pub unsafe extern "C" fn aff_sam_contains_ngram(
    sam: *const AffSam,
    ngram_ptr: *const u32,
    ngram_len: usize,
) -> bool {
    if sam.is_null() || ngram_ptr.is_null() || ngram_len == 0 {
        return false;
    }
    let ng = slice::from_raw_parts(ngram_ptr, ngram_len);
    (*sam).contains_ngram(ng)
}

#[no_mangle]
pub unsafe extern "C" fn aff_sam_first_endpos(
    sam: *const AffSam,
    ngram_ptr: *const u32,
    ngram_len: usize,
) -> usize {
    if sam.is_null() || ngram_ptr.is_null() || ngram_len == 0 {
        return 0;
    }
    let ng = slice::from_raw_parts(ngram_ptr, ngram_len);
    (*sam).first_endpos(ng)
}

#[no_mangle]
pub unsafe extern "C" fn aff_sam_matches_prefix(
    sam: *const AffSam,
    ptr: *const u32,
    len: usize,
) -> bool {
    if sam.is_null() {
        return false;
    }
    let tokens = &(*sam).tokens;
    if len > tokens.len() {
        return false;
    }
    if len == 0 {
        return true;
    }
    if ptr.is_null() {
        return false;
    }
    let s = slice::from_raw_parts(ptr, len);
    tokens[..len] == *s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_online_push_and_unbounded_match_len_24() {
        let mut sam = AffSam::new();
        let pat: Vec<u32> = (1000..1024).collect(); // 24 tokens
        let cont: Vec<u32> = (2000..2012).collect();

        sam.push_slice(&pat);
        sam.push_slice(&cont);
        sam.push_slice(&[9999, 9998]);
        sam.push_slice(&pat);

        assert_eq!(sam.len(), 24 + 12 + 2 + 24);
        assert_eq!(sam.match_len(), 24);

        let mut out = [0u32; 16];
        let prop = sam.propose_impl(&sam.tokens, 12, 4, 0.5, &mut out);
        assert_eq!(prop.match_len, 24);
        assert_eq!(prop.len, 12);
        assert!((prop.score - 1.0).abs() < 1e-6);
        assert!((prop.expect - 12.0).abs() < 1e-5);
        assert_eq!(&out[..12], &cont[..]);
    }

    #[test]
    fn test_non_destructive_rollback_preserves_prefix() {
        let mut sam = AffSam::new();
        let prefix: Vec<u32> = vec![
            10, 11, 12, 13, 14, 15, 16, 17, 50, 51, 52, 53,
            90, 91, 92, 10, 11, 12, 13, 14, 15, 16, 17,
        ];
        sam.push_slice(&prefix);
        assert_eq!(sam.match_len(), 8);
        let base_len = sam.len();

        // Speculate / append extra turn tail with diverging branch, then roll back.
        sam.push_slice(&[60, 61, 62, 63, 10, 11, 12, 13, 14, 15, 16, 17, 70, 71]);
        assert!(sam.len() > base_len);
        assert!(sam.contains_ngram(&[60, 61, 62, 63]));

        assert!(!sam.rollback_to(sam.len() + 1));
        assert!(sam.rollback_to(base_len));
        assert_eq!(sam.len(), base_len);
        assert_eq!(sam.match_len(), 8);
        assert!(!sam.contains_ngram(&[60, 61, 62, 63]));
        assert!(sam.contains_ngram(&[50, 51, 52, 53]));

        // Fresh SAM built on `prefix` alone must match rolled-back SAM identically.
        let mut fresh = AffSam::new();
        fresh.push_slice(&prefix);
        assert_eq!(sam.match_len(), fresh.match_len());

        let mut out_rb = [0u32; 8];
        let mut out_fr = [0u32; 8];
        let prb = sam.propose_impl(&prefix, 4, 4, 0.5, &mut out_rb);
        let pfr = fresh.propose_impl(&prefix, 4, 4, 0.5, &mut out_fr);
        assert_eq!(prb.len, pfr.len);
        assert_eq!(prb.match_len, pfr.match_len);
        assert_eq!(prb.score, pfr.score);
        assert_eq!(prb.expect, pfr.expect);
        assert_eq!(&out_rb[..prb.len], &out_fr[..pfr.len]);
        assert_eq!(&out_rb[..prb.len], &[50, 51, 52, 53]);

        // Continuing after rollback also records new stats cleanly.
        sam.push_slice(&[50, 51, 52, 53, 888]);
        assert!(sam.contains_ngram(&[50, 51, 52, 53, 888]));
    }

    #[test]
    fn test_proposal_confidence_and_branching() {
        let mut sam = AffSam::new();
        let seq = vec![
            1001, 1002, 1003, 10, 11, 12, 13, 21, 2001,
            1004, 1005, 1006, 10, 11, 12, 13, 20, 30, 2002,
            1007, 1008, 1009, 10, 11, 12, 13, 20, 30, 2003,
            1010, 1011, 1012, 10, 11, 12, 13,
        ];
        sam.push_slice(&seq);

        let mut out = [0u32; 8];
        let hi = sam.propose_impl(&seq, 4, 4, 0.8, &mut out);
        assert_eq!(hi.len, 0);

        let lo = sam.propose_impl(&seq, 4, 4, 0.5, &mut out);
        assert_eq!(lo.len, 2);
        assert_eq!(lo.match_len, 4);
        assert_eq!(&out[..2], &[20, 30]);
        assert!((lo.score - (2.0 / 3.0)).abs() < 1e-5);
        assert!((lo.expect - (4.0 / 3.0)).abs() < 1e-5);

        // External context query (not matching tail of sam.tokens) also works.
        let ext_ctx = [10u32, 11, 12, 13];
        let ext = sam.propose_impl(&ext_ctx, 4, 4, 0.5, &mut out);
        assert_eq!(ext.len, 2);
        assert_eq!(ext.match_len, 4);
        assert_eq!(&out[..2], &[20, 30]);
    }

    #[test]
    fn test_contains_ngram_positive_negative_and_first_endpos() {
        unsafe {
            let sam = aff_sam_new();
            let prompt = [1u32, 2, 3, 4, 5, 6, 7, 8, 9];
            aff_sam_push_slice(sam, prompt.as_ptr(), prompt.len());
            aff_sam_push_slice(sam, [100, 101, 102, 103, 104].as_ptr(), 5);

            assert!(aff_sam_contains_ngram(sam, [3, 4, 5, 6].as_ptr(), 4));
            assert!(aff_sam_contains_ngram(sam, [100, 101, 102].as_ptr(), 3));
            assert!(!aff_sam_contains_ngram(sam, [3, 4, 5, 99].as_ptr(), 4));
            assert!(!aff_sam_contains_ngram(sam, [1].as_ptr(), 0));

            let ep_prompt = aff_sam_first_endpos(sam, [3, 4, 5, 6].as_ptr(), 4);
            let ep_gen = aff_sam_first_endpos(sam, [100, 101, 102].as_ptr(), 3);
            assert_eq!(ep_prompt, 6);
            assert_eq!(ep_gen, 12);

            let mut out_toks = [0u32; 8];
            let mut out_len = 0usize;
            let mut out_score = 0.0f32;
            let mut out_match = 0usize;
            aff_sam_push_slice(sam, [2, 3, 4, 5].as_ptr(), 4);
            let ok = aff_sam_propose(
                sam,
                [2u32, 3, 4, 5].as_ptr(),
                4,
                4,
                4,
                0.5,
                out_toks.as_mut_ptr(),
                &mut out_len,
                &mut out_score,
                &mut out_match,
            );
            assert!(ok);
            assert_eq!(out_len, 4);
            assert_eq!(out_match, 4);
            assert_eq!(out_toks[..4], [6, 7, 8, 9]);

            aff_sam_free(sam);
        }
    }

    #[test]
    fn test_50k_token_benchmark_and_periodic_stress() {
        // Mixed high-entropy + repeated-code 50K stream.
        let mut sam = AffSam::new();
        let mut seq = Vec::with_capacity(50_000);
        let mut rng: u64 = 0xdead_beef_cafe_babe;
        for i in 0..50_000usize {
            if (i % 512) < 32 {
                seq.push(70_000 + (i % 32) as u32);
            } else {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                seq.push(((rng >> 32) as u32) % 60_000);
            }
        }
        // Append 16-token repeated anchor at end so match_len == 16.
        seq.extend(70_000..70_016);
        let t0 = Instant::now();
        sam.push_slice(&seq);
        let dt = t0.elapsed();
        assert_eq!(sam.len(), 50_016);
        assert!(sam.contains_ngram(&[70_000, 70_001, 70_002, 70_003]));
        assert!(sam.contains_ngram(&[70_028, 70_029, 70_030, 70_031]));
        assert!(!sam.contains_ngram(&[70_000, 70_001, 70_002, 70_031]));

        let mut out = [0u32; 16];
        let prop = sam.propose_impl(&seq, 12, 4, 0.5, &mut out);
        assert_eq!(prop.match_len, 16);
        assert_eq!(prop.len, 12);
        assert_eq!(&out[..12], &(70_016..70_028).collect::<Vec<u32>>()[..]);
        eprintln!(
            "rust_sam 50k mixed push: {:.2} ms, states={}",
            dt.as_secs_f64() * 1e3,
            sam.states.len()
        );
        assert!(dt.as_secs_f64() < 1.0);

        // Adversarial 50K identical-token periodic stress must stay bounded O(1)/push.
        let mut periodic = AffSam::new();
        let t1 = Instant::now();
        for _ in 0..50_000 {
            periodic.push(42);
        }
        let dt_per = t1.elapsed();
        assert_eq!(periodic.len(), 50_000);
        assert_eq!(periodic.match_len(), 49_999);
        let p_per = periodic.propose_impl(&[42, 42, 42, 42], 8, 4, 0.5, &mut out);
        assert_eq!(p_per.len, 8);
        eprintln!(
            "rust_sam 50k periodic push: {:.2} ms",
            dt_per.as_secs_f64() * 1e3
        );
        assert!(dt_per.as_secs_f64() < 1.0);
    }

    #[test]
    fn test_rollback_across_edge_map_promotion_and_multi_branch_stats() {
        let mut sam = AffSam::new();
        // Anchor [10, 11, 12, 13] followed by 7 distinct continuations so state promotes
        // from inline_edges (cap=4) to SmallEdgeHash.
        for c in 100u32..104 {
            sam.push_slice(&[900 + c, 10, 11, 12, 13, c]);
        }
        let pre_promote_len = sam.len();

        // Add 5th, 6th, 7th branches triggering SmallEdgeHash promotion and hash updates.
        for c in 104u32..107 {
            sam.push_slice(&[900 + c, 10, 11, 12, 13, c]);
        }
        assert!(sam.contains_ngram(&[10, 11, 12, 13, 104]));
        assert!(sam.contains_ngram(&[10, 11, 12, 13, 106]));

        // Roll back across the promotion boundary back to 4 branches.
        assert!(sam.rollback_to(pre_promote_len));
        assert_eq!(sam.len(), pre_promote_len);
        assert!(sam.contains_ngram(&[10, 11, 12, 13, 103]));
        assert!(!sam.contains_ngram(&[10, 11, 12, 13, 104]));
        assert!(!sam.contains_ngram(&[10, 11, 12, 13, 106]));

        // Now push a 3rd+ dominant continuation token 200 many times and verify conservative stats swap.
        for k in 0u32..10 {
            sam.push_slice(&[1100 + k, 10, 11, 12, 13, 200, 201]);
        }
        sam.push_slice(&[1200, 10, 11, 12, 13]);
        let mut out = [0u32; 4];
        let prop = sam.propose_impl(&sam.tokens, 2, 4, 0.5, &mut out);
        assert_eq!(prop.len, 2);
        assert_eq!(prop.match_len, 4);
        assert_eq!(&out[..2], &[200, 201]);
        assert!(prop.score > 0.5 && prop.score <= 1.0);
    }
}
