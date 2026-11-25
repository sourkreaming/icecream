//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Blocks found
//! wholly uncommitted are skipped while they remain so; one query can
//! confirm a whole region, including ordinary reservations between views.
//!
//! Writes into physical pages through another view are seen too: the
//! runtime tells us the views of the pages it wrote itself ([`writing`],
//! xmem's `write_physical` and DirectStorage's page destinations), and a
//! write by the title through any view of a guarded page faults like one
//! at the guarded address (the guard makes every view of the page
//! read-only, [`open`] gives them all back; not a page of more than
//! [`MANY_VIEWS`] views). A
//! system call writing into a guarded block fails (none is known to).

use std::collections::BTreeMap;
use std::sync::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::sync::atomic::{AtomicU64, Ordering};

/// Disjoint, non-touching half-open intervals `[lo, hi)` (`lo < hi`), stored
/// as `lo -> hi`. Lets [`Guards`] check a 174 MiB clean view heap in one
/// `BTreeMap` lookup instead of walking thousands of 64 KiB block entries.
#[derive(Default)]
struct RangeSet(BTreeMap<usize, usize>);

impl RangeSet {
    const fn new() -> Self {
        Self(BTreeMap::new())
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn contains(&self, x: usize) -> bool {
        self.0
            .range(..=x)
            .next_back()
            .is_some_and(|(_, &hi)| x < hi)
    }

    fn contains_range(&self, lo: usize, hi: usize) -> bool {
        lo >= hi
            || self
                .0
                .range(..=lo)
                .next_back()
                .is_some_and(|(_, &end)| end >= hi)
    }

    fn insert(&mut self, x: usize) {
        if let Some(hi) = x.checked_add(1) {
            self.insert_range(x, hi);
        }
    }

    fn insert_range(&mut self, mut lo: usize, mut hi: usize) {
        if lo >= hi {
            return;
        }
        if let Some((&p_lo, &p_hi)) = self.0.range(..=lo).next_back()
            && p_hi >= lo
        {
            lo = p_lo;
            hi = hi.max(p_hi);
        }
        let merge: Vec<(usize, usize)> = self.0.range(lo..=hi).map(|(&a, &b)| (a, b)).collect();
        for (a, b) in merge {
            self.0.remove(&a);
            hi = hi.max(b);
        }
        self.0.insert(lo, hi);
    }

    fn remove_range(&mut self, lo: usize, hi: usize) {
        if lo >= hi {
            return;
        }
        if let Some((&p_lo, &p_hi)) = self.0.range(..lo).next_back()
            && p_hi > lo
        {
            self.0.insert(p_lo, lo);
            if p_hi > hi {
                self.0.insert(hi, p_hi);
                return;
            }
        }
        let overlap: Vec<(usize, usize)> = self.0.range(lo..hi).map(|(&a, &b)| (a, b)).collect();
        for (a, b) in overlap {
            self.0.remove(&a);
            if b > hi {
                self.0.insert(hi, b);
            }
        }
    }

    fn covered_in(&self, lo: usize, hi: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        if lo >= hi {
            return out;
        }
        if let Some((&a, &b)) = self.0.range(..lo).next_back()
            && b > lo
        {
            let _ = a;
            out.push((lo, b.min(hi)));
        }
        for (&a, &b) in self.0.range(lo..hi) {
            out.push((a, b.min(hi)));
        }
        out
    }

    fn uncovered_in(&self, lo: usize, hi: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        if lo >= hi {
            return out;
        }
        let mut cur = lo;
        if let Some((_, &b)) = self.0.range(..=lo).next_back()
            && b > cur
        {
            cur = b.min(hi);
        }
        for (&a, &b) in self.0.range(lo..hi) {
            if a > cur {
                out.push((cur, a.min(hi)));
            }
            cur = cur.max(b).min(hi);
            if cur >= hi {
                break;
            }
        }
        if cur < hi {
            out.push((cur, hi));
        }
        out
    }

    fn count_in(&self, lo: usize, hi: usize) -> usize {
        if lo >= hi {
            return 0;
        }
        let mut total = 0usize;
        if let Some((_, &b)) = self.0.range(..lo).next_back()
            && b > lo
        {
            total += b.min(hi) - lo;
        }
        for (&a, &b) in self.0.range(lo..hi) {
            total += b.min(hi) - a;
        }
        total
    }

    fn covered_end(&self, x: usize) -> usize {
        self.0
            .range(..=x)
            .next_back()
            .filter(|&(_, &hi)| x < hi)
            .map_or(x, |(_, &hi)| hi)
    }

    fn overlaps(&self, lo: usize, hi: usize) -> bool {
        if lo >= hi {
            return false;
        }
        if let Some((_, &b)) = self.0.range(..lo).next_back()
            && b > lo
        {
            return true;
        }
        self.0.range(lo..hi).next().is_some()
    }
}

/// Bumped whenever any guarded block is opened, remapped, or aliased without
/// being clean: invalidates cached guarded ranges in [`Journal::ahead_guarded`].
static GUARD_GEN: AtomicU64 = AtomicU64::new(0);

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

/// A title-memory range whose changed pages another subsystem needs to read.
pub(crate) struct Tracked {
    start: usize,
    end: usize,
    seen: Mutex<BTreeMap<usize, u64>>,
}

/// Tracks a runtime GPU-memory range, including physical-page views when
/// `views` is true. Other process memory is not tracked.
pub(crate) fn track(start: usize, len: usize, views: bool) -> Option<Tracked> {
    let end = start.checked_add(len)?;
    if len == 0 || writes(start, end, views).is_none() {
        return None;
    }
    Some(Tracked {
        start,
        end,
        seen: Mutex::default(),
    })
}

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// The highest asking number recorded in `written` (0: none yet).
    last_write: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// Page ranges asked ahead for a submission's refreshes ([`ask_ahead`])
    /// and the asking that did: `writes` answers for a range inside one
    /// from the journal, without another kernel call.
    ahead: Vec<(usize, usize)>,
    /// Guarded view-memory page ranges checked ahead for the submission,
    /// valid while `GUARD_GEN == ahead_guard_gen`.
    ahead_guarded: Vec<(usize, usize)>,
    ahead_guard_gen: u64,
    ahead_asked: u64,
    /// The ahead asking stands for a whole submission ([`ask_once`]): no
    /// other ahead asking is made, and `ahead` stays, until
    /// [`ask_once_end`].
    kept: bool,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    last_write: 0,
    written: BTreeMap::new(),
    found: Vec::new(),
    ahead: Vec::new(),
    ahead_guarded: Vec::new(),
    ahead_guard_gen: 0,
    ahead_asked: 0,
    kept: false,
});

/// Ranges of one submission closer than this are asked about together:
/// walking the pages between them costs less than another kernel call (a
/// call that resets written pages costs 30-70 microseconds whatever the
/// range).
const AHEAD_GAP: usize = (8 << 20) / PAGE;
/// The most one asking covers (its answer needs 8 bytes per 4 KiB page).
const AHEAD_MAX: usize = (1 << 30) / PAGE;

/// Asks the kernel about many ranges `(start, end)` at once, before the
/// [`writes`] of each: ranges close together are one asking, and `writes`
/// then answers from the journal until [`ask_ahead_done`]. The world's
/// submissions refresh thousands of host copies (6,664 askings a frame,
/// 456 ms, in run 14). Ranges that are not write-watched, or lie alone, are
/// left to `writes`.
pub(crate) fn ask_ahead(ranges: &[(usize, usize)]) {
    let mut pieces: Vec<(usize, usize)> = ranges
        .iter()
        .filter(|r| r.0 < r.1)
        .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
        .collect();
    if pieces.len() < 2 {
        return;
    }
    pieces.sort_unstable();
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // The submission's one asking stands ([`ask_once`]): what it does not
    // hold is left to `writes`.
    if j.kept {
        return;
    }
    j.asked += 1;
    j.ahead_asked = j.asked;
    j.ahead.clear();
    let mut i = 0;
    while i < pieces.len() {
        let first = pieces[i].0;
        let mut last = pieces[i].1;
        let mut k = i + 1;
        while k < pieces.len()
            && pieces[k].0 <= last + AHEAD_GAP
            && last.max(pieces[k].1) - first <= AHEAD_MAX
        {
            last = last.max(pieces[k].1);
            k += 1;
        }
        ask_run(j, &pieces[i..k]);
        i = k;
    }
}

/// Since the last frame line: the kernel askings that took a millisecond
/// or more, their microseconds, and the longest asking's. An asking walks
/// page tables, some ten microseconds for most ranges; in run 75 the 2,644
/// askings of a frame took 219 ms in the lines where the pass over the
/// heap's views ran long, 25-50 ms in the others. These numbers say
/// whether a few askings wait for something or all of them are slow.
#[cfg(feature = "diagnostics")]
static SLOW: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(feature = "diagnostics")]
fn slow(us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if us >= 1000 {
        SLOW[0].fetch_add(1, Relaxed);
        SLOW[1].fetch_add(us, Relaxed);
    }
    SLOW[2].fetch_max(us, Relaxed);
}

/// [`SLOW`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn slow_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| SLOW[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// The submission's refreshes are done: `writes` asks the kernel again.
pub(crate) fn ask_ahead_done() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if !j.kept {
        j.ahead.clear();
    }
}

/// Whether a submission asks the kernel once, before its first refresh,
/// about every range its lists refresh (`VICTORY_ASK_ONCE=1`; off by
/// default).
///
/// A submission's refreshes are done in groups, between the title's own
/// buffer copies (each copy flushes the refreshes recorded before it), and
/// every group asked the kernel again: 2,644 askings a frame, 70 a
/// submission. With the switch the journal answers all of them from one
/// asking made when the submission starts.
///
/// What changes: memory the title's other threads write while the
/// submission is being prepared is seen by the next submission's asking,
/// not by this one's later groups. What this submission's lists read was
/// written before they were submitted, so before the asking. Our own
/// writes into title memory while preparing are told to the journal
/// ([`wrote`]) or end the asking ([`ask_once_end`]).
pub(crate) fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default since run 81 (with card buffers: 375-450 ms a frame
    // against 573-647, the picture unchanged over four minutes);
    // `VICTORY_ASK_ONCE=0` asks per range, as before.
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE").is_none_or(|v| v != "0"))
}

/// With [`once`]: asks about `ranges` (everything a submission's lists
/// refresh) as [`ask_ahead`] does, and lets that asking stand until
/// [`ask_once_end`]. Ranges already tracked as guarded view memory are
/// checked through [`Guards`] once instead of hitting failing
/// `GetWriteWatch` syscalls, and cached until [`GUARD_GEN`] advances.
pub(crate) fn ask_once(ranges: &[(usize, usize)]) {
    if !once() {
        return;
    }
    ask_once_end();
    // Split out known guarded view-memory ranges before calling GetWriteWatch,
    // avoiding thousands of failing kernel syscalls per frame.
    let mut other = Vec::new();
    let mut guarded_ranges = Vec::new();
    {
        let ask = ask_view();
        let g = guards_read();
        for &(start, end) in ranges {
            if start >= end {
                continue;
            }
            let first = start / PAGE;
            if !ask && g.knows(first) {
                guarded_ranges.push((first, end.div_ceil(PAGE)));
            } else {
                other.push((start, end));
            }
        }
    }
    if other.len() >= 2 {
        ask_ahead(&other);
    }
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    if j.ahead.is_empty() {
        // `ask_ahead` asked nothing (fewer than two write-watched ranges).
        j.asked += 1;
        j.ahead_asked = j.asked;
    }
    // `ask_ahead` leaves a range that lies alone to `writes`, which asks
    // once per flush: with one asking for the submission that was every
    // such range asked again at each of its 70 flushes (run 76: 2,300
    // askings a frame with the switch on, as without it). They are asked
    // here, once; what is not write-watched stays with `writes`.
    for &(start, end) in &other {
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        if !j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
            if ask_kernel(j, first, last) {
                j.ahead.push((first, last));
            } else {
                guarded_ranges.push((first, last));
            }
        }
    }
    if !guarded_ranges.is_empty() {
        // Merge overlapping/touching view ranges so one guard check covers all.
        guarded_ranges.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(guarded_ranges.len());
        for (a, b) in guarded_ranges {
            match merged.last_mut() {
                Some(last) if a <= last.1 => last.1 = last.1.max(b),
                _ => merged.push((a, b)),
            }
        }
        j.ahead_guarded.clear();
        let gen_before = GUARD_GEN.load(Ordering::Acquire);
        let now = j.ahead_asked;
        for (first, last) in merged {
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            let guarding = std::time::Instant::now();
            let written = guard(first * PAGE, last * PAGE);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
            let Some(written) = written else {
                continue;
            };
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::Watches, 1);
                add(C::WatchUs, started.elapsed().as_micros() as u64);
                slow(started.elapsed().as_micros() as u64);
            }
            if !written.is_empty() {
                for page in written {
                    j.written.insert(page, now);
                }
                j.last_write = j.last_write.max(now);
            }
            j.ahead_guarded.push((first, last));
        }
        // Only keep the guarded cache if no concurrent fault/remap happened
        // while checking the merged ranges.
        if GUARD_GEN.load(Ordering::Acquire) == gen_before {
            j.ahead_guard_gen = gen_before;
        } else {
            j.ahead_guarded.clear();
        }
    }
    j.kept = !j.ahead.is_empty() || !j.ahead_guarded.is_empty();
}

/// The submission's one asking stands no longer: its refreshes are done,
/// or title memory was written in a way the journal was not told.
pub(crate) fn ask_once_end() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if j.kept {
        j.kept = false;
        j.ahead.clear();
        j.ahead_guarded.clear();
    }
}

/// We wrote title memory `[at, at + len)` while a submission's one asking
/// stands (the title's buffer copies, done at submission): its pages count
/// as written after that asking, so the refreshes that follow copy them.
pub(crate) fn wrote(at: usize, len: usize) {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if !j.kept || len == 0 {
        return;
    }
    j.asked += 1;
    let now = j.asked;
    // The asking that stands moves with it: `asked` and `writes` give it as
    // "now", and a page noted written after "now" stays stale for every
    // copy that takes it (copied at `now`, written at `now + 1`): copied
    // again at every refresh, and a reason to settle flights
    // (fxceless/victory-fork#410).
    j.ahead_asked = now;
    for page in at / PAGE..at.saturating_add(len).div_ceil(PAGE) {
        j.written.insert(page, now);
    }
    j.last_write = j.last_write.max(now);
}

/// One kernel asking for `pieces` (page ranges in order); by halves if it
/// fails, as a run across two reservations or into memory that is not
/// write-watched does. A piece alone is left to [`writes`].
fn ask_run(j: &mut Journal, pieces: &[(usize, usize)]) {
    if pieces.len() < 2 {
        return;
    }
    let first = pieces[0].0;
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    if ask_kernel(j, first, last) {
        j.ahead.push((first, last));
        return;
    }
    let (a, b) = pieces.split_at(pieces.len() / 2);
    ask_run(j, a);
    ask_run(j, b);
}

/// Asks the kernel for the written pages of pages `[first, last)` and notes
/// them in the journal under the ahead asking; false (and nothing changed)
/// if the range is not write-watched memory of one reservation.
fn ask_kernel(j: &mut Journal, first: usize, last: usize) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    if failed {
        return false;
    }
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    let now = j.ahead_asked;
    if count > 0 {
        for k in 0..count {
            let page = j.found[k] / PAGE;
            j.written.insert(page, now);
        }
        j.last_write = j.last_write.max(now);
    }
    true
}

/// Pages of `[first, last)` in `j.written` written strictly after `since`.
/// Returns immediately without walking `j.written` when `j.last_write <= since`.
fn collect_since(j: &Journal, first: usize, last: usize, since: u64) -> Vec<(usize, u64)> {
    if since > 0 && j.last_write <= since {
        return Vec::new();
    }
    j.written
        .range(first..last)
        .filter(|&(_, &w)| w > since)
        .map(|(&p, &w)| (p, w))
        .collect()
}

/// What [`writes`] would say of `[start, end)` from the journal alone, if
/// the range lies in one asked ahead ([`ask_ahead`], [`ask_once`]): the
/// pages written since tracking began and the asking that last found each
/// written, with the number of the asking that stands (as [`writes`] gives
/// it). `None` if the kernel would have to be asked.
pub(crate) fn asked(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let in_ahead = j.ahead.iter().any(|&(a, b)| a <= first && last <= b);
    let in_guarded = !in_ahead
        && !j.ahead_guarded.is_empty()
        && GUARD_GEN.load(Ordering::Acquire) == j.ahead_guard_gen
        && j.ahead_guarded.iter().any(|&(a, b)| a <= first && last <= b);
    (in_ahead || in_guarded).then(|| (j.ahead_asked, collect_since(&j, first, last, 0)))
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    writes_since_kind(start, end, guarded, 0).map(|(now, written, _)| (now, written))
}

/// Like [`writes`], returning only pages written strictly after `since`.
pub(crate) fn writes_since(
    start: usize,
    end: usize,
    guarded: bool,
    since: u64,
) -> Option<(u64, Vec<(usize, u64)>)> {
    writes_since_kind(start, end, guarded, since).map(|(now, written, _)| (now, written))
}

/// Like [`writes_since`], also returning whether the range was tracked by
/// view-memory guards (`true`) rather than kernel write watching (`false`).
pub(crate) fn writes_since_kind(
    start: usize,
    end: usize,
    allow_guard: bool,
    since: u64,
) -> Option<(u64, Vec<(usize, u64)>, bool)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // Asked ahead with the submission's other ranges: the journal has it.
    if j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
        return Some((j.ahead_asked, collect_since(j, first, last, since), false));
    }
    if allow_guard && !j.ahead_guarded.is_empty() {
        if GUARD_GEN.load(Ordering::Acquire) != j.ahead_guard_gen {
            j.ahead_guarded.clear();
        } else if j.ahead_guarded.iter().any(|&(a, b)| a <= first && last <= b) {
            return Some((j.ahead_asked, collect_since(j, first, last, since), true));
        }
    }
    // Fast path for already-known guarded view memory: skip the failing
    // `GetWriteWatch` kernel syscall entirely. Try a shared read lock first
    // so clean view checks never contend with other readers.
    if allow_guard && !ask_view() {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let known_written = {
            let clean_hit = {
                let g = guards_read();
                if g.knows(first) {
                    Some(g.clean.contains_range(first, last))
                } else {
                    None
                }
            };
            match clean_hit {
                Some(true) => Some(Vec::new()),
                Some(false) => {
                    let mut g = guards();
                    if g.knows(first) {
                        #[cfg(feature = "diagnostics")]
                        let guarding = std::time::Instant::now();
                        let w = guard_locked(&mut g, first, last);
                        #[cfg(feature = "diagnostics")]
                        crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
                        Some(w)
                    } else {
                        None
                    }
                }
                None => None,
            }
        };
        if let Some(written) = known_written {
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::WatchKnownView, 1);
                add(C::Watches, 1);
                add(C::WatchUs, started.elapsed().as_micros() as u64);
                slow(started.elapsed().as_micros() as u64);
            }
            j.asked += 1;
            let now = j.asked;
            if !written.is_empty() {
                for page in written {
                    j.written.insert(page, now);
                }
                j.last_write = j.last_write.max(now);
            }
            return Some((now, collect_since(j, first, last, since), true));
        }
    }
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::WatchFailed, failed as u64);
    let was_guarded = failed;
    let guarded = if failed {
        count = 0;
        if !allow_guard {
            return None;
        }
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        let written = guard(first * PAGE, last * PAGE);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
        written?
    } else {
        Vec::new()
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    j.asked += 1;
    let now = j.asked;
    if count > 0 || !guarded.is_empty() {
        for k in 0..count {
            let page = j.found[k] / PAGE;
            j.written.insert(page, now);
        }
        for page in guarded {
            j.written.insert(page, now);
        }
        j.last_write = j.last_write.max(now);
    }
    Some((now, collect_since(j, first, last, since), was_guarded))
}

impl Tracked {
    /// Takes changed pages of `[from, to)`, cut to the tracked range. A
    /// partial page stays changed so a later take can read its other bytes.
    pub(crate) fn take(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from >= to {
            return Vec::new();
        }
        let Some((_, pages)) = writes(from, to, true) else {
            return Vec::new();
        };
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for (page, write) in pages {
            if seen.get(&page).is_some_and(|&last| last >= write) {
                continue;
            }
            let (start, end) = ((page * PAGE).max(from), ((page + 1) * PAGE).min(to));
            if start >= end {
                continue;
            }
            // Keep partial pages dirty: the caller has not taken the rest.
            if (page * PAGE).max(self.start) >= from
                && ((page + 1) * PAGE).min(self.end) <= to
            {
                seen.insert(page, write);
            }
            match ranges.last_mut() {
                Some((_, last)) if *last == start => *last = end,
                _ => ranges.push((start, end)),
            }
        }
        ranges
    }
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

/// Whether to still ask `GetWriteWatch` on memory already tracked as view
/// memory (`VICTORY_VIEW_ASK=1` or `VICTORY_GUARD_ASK=1`).
fn ask_view() -> bool {
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ASK.get_or_init(|| {
        ["VICTORY_VIEW_ASK", "VICTORY_GUARD_ASK"]
            .iter()
            .any(|n| std::env::var_os(n).is_some_and(|v| v == "1"))
    })
}

struct Guards {
    /// Blocks we made read-only.
    protected: RangeSet,
    /// Blocks that need no work on the next check: protected blocks not
    /// written since a copy took them, plus unmapped placeholder/free holes
    /// ([`Self::holes`]) that cannot become mapped without
    /// [`title_memory_remapped`].
    clean: RangeSet,
    /// Reserved view blocks (`MEM_MAPPED` + `MEM_RESERVE`) copied as zeros.
    /// Ordinary commits need no runtime mapping notification, so these are
    /// re-checked on each asking rather than added to `clean`.
    absent: RangeSet,
    /// Unmapped placeholder or free holes between views copied as zeros.
    /// They cannot be committed in place without `xmem` mapping a view and
    /// calling [`title_memory_remapped`], so they are also placed in `clean`
    /// and skipped in O(log intervals) with no `VirtualQuery` syscalls.
    holes: RangeSet,
    /// Blocks whose physical page was guarded together with its other views
    /// (those views' blocks are in it too): a write to one opens them all.
    /// A write range can be huge (`fread`), so only these are looked up.
    aliased: RangeSet,
}

impl Guards {
    fn knows(&self, block: usize) -> bool {
        self.protected.contains(block)
            || self.absent.contains(block)
            || self.holes.contains(block)
    }

    fn overlaps_any(&self, lo: usize, hi: usize) -> bool {
        self.protected.overlaps(lo, hi)
            || self.absent.overlaps(lo, hi)
            || self.holes.overlaps(lo, hi)
    }
}

static GUARDS: RwLock<Guards> = RwLock::new(Guards {
    protected: RangeSet::new(),
    clean: RangeSet::new(),
    absent: RangeSet::new(),
    holes: RangeSet::new(),
    aliased: RangeSet::new(),
});

fn guards_read() -> RwLockReadGuard<'static, Guards> {
    GUARDS.read().unwrap_or_else(|e| e.into_inner())
}

fn guards() -> RwLockWriteGuard<'static, Guards> {
    GUARDS.write().unwrap_or_else(|e| e.into_inner())
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    // A range whose first block the guards hold already (read-only by our
    // hand, or noted absent/hole) is a view: no question to Windows. The
    // question walks the page tables to the end of the pages that share the
    // first one's state, and a guarded view is one long run of read-only
    // pages: 0.6 ms for the title's 174 MB heap, 16 times a frame, and most
    // of the 68 ms a frame of "guards" (run 58).
    // `VICTORY_GUARD_ASK=1` asks every time, as before.
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    let ask = *ASK.get_or_init(|| std::env::var_os("VICTORY_GUARD_ASK").is_some_and(|v| v == "1"));
    if !ask {
        let hit = {
            let g = guards_read();
            g.knows(first).then(|| g.clean.contains_range(first, last))
        };
        match hit {
            Some(true) => return Some(Vec::new()),
            Some(false) => {
                let mut g = guards();
                if g.knows(first) {
                    return Some(guard_locked(&mut g, first, last));
                }
            }
            None => {}
        }
    }
    if !is_view(start, end) {
        return None;
    }
    install();
    let mut g = guards();
    Some(guard_locked(&mut g, first, last))
}

/// [`guard`] with `g` already held and the handler already installed.
fn guard_locked(g: &mut Guards, first: usize, last: usize) -> Vec<usize> {
    // Fast path: the entire range is already clean (O(log intervals), zero
    // allocation). Nearly every refresh in the world hits this.
    if g.clean.contains_range(first, last) {
        return Vec::new();
    }
    let mut written = Vec::new();
    for (gap_lo, gap_hi) in g.clean.uncovered_in(first, last) {
        let mut block = gap_lo;
        let mut uncommitted_until = gap_lo;
        let mut run: Option<usize> = None;
        while block < gap_hi {
            if g.absent.contains(block) {
                // Ordinary commits need no runtime mapping notification. Reuse
                // this asking's query for the following blocks in the same region.
                if block >= uncommitted_until {
                    uncommitted_until = uncommitted_end(block);
                }
                if block < uncommitted_until {
                    if let Some(from) = run.take() {
                        protect(g, from, block);
                    }
                    let next = g.absent.covered_end(block).min(uncommitted_until).min(gap_hi);
                    block = next.max(block + 1);
                    continue;
                }
                g.absent.remove_range(block, block + 1);
            }
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if g.protected.contains(block) {
                if let Some(from) = run.take() {
                    protect(g, from, block);
                }
                // Written through the runtime or aliased, still read-only.
                g.clean.insert(block);
            } else {
                run.get_or_insert(block);
            }
            block += 1;
        }
        if let Some(from) = run.take() {
            protect(g, from, gap_hi);
        }
    }
    written
}

/// Whether `[start, end)` is view memory. The title's views lie in
/// placeholders, mapped and unmapped block by block as it streams, so a
/// range can begin in a block that is not mapped right now: the first
/// pages of the range that are there decide. A range with nothing there
/// counts as a view (its blocks are noted absent, and looked at again when
/// the runtime maps them).
///
/// Looking at the first address only took such a range for "not a view":
/// its copy was marked not watched for good and copied whole at every
/// refresh, around its unmapped pages (run 54: 300 such ranges, 570 MB
/// copied a frame for them, 255 copies a frame reading around holes).
/// `VICTORY_GUARD_HOLES=0` looks at the first address only, as before.
fn is_view(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    static HOLES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let holes = *HOLES.get_or_init(|| std::env::var_os("VICTORY_GUARD_HOLES").is_none_or(|v| v != "0"));
    let mut at = start;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return false;
        }
        if info.Type == MEM_MAPPED {
            return true;
        }
        // Committed memory that is no view (the title's heaps, ours), or
        // the answer as it was before.
        if info.State == MEM_COMMIT || !holes {
            return false;
        }
        // A hole (a placeholder, or free): on to what follows it.
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= at || next >= end {
            return true;
        }
        at = next;
    }
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail, and
/// noting those not committed (unmapped) absent. The same physical pages
/// at every other view of them are made read-only too: a write through
/// one of those is this memory written, and only a fault can say so.
/// `VICTORY_VIEW_GUARD=0` leaves the other views as they are, as before.
fn protect(g: &mut Guards, from: usize, to: usize) {
    protect_here(g, from, to);
    if !view_guard() {
        return;
    }
    for (block, shown) in
        crate::inplace::aliased_pages_in(from.saturating_mul(BLOCK), to.saturating_mul(BLOCK), MANY_VIEWS)
    {
        let block = block / BLOCK;
        #[cfg(feature = "diagnostics")]
        ALIASES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        for alias in shown {
            let alias = alias / BLOCK;
            g.aliased.insert(alias);
            if alias != block {
                protect_alias(g, alias);
            }
        }
    }
}

/// Views of one page past which its views are not guarded together.
pub(crate) const MANY_VIEWS: usize = 64;

/// Whether all the views of a guarded page are guarded together
/// (`VICTORY_VIEW_GUARD`).
fn view_guard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_GUARD").is_none_or(|v| v != "0"))
}

/// Diagnostics: blocks guarded together with their other views, blocks
/// opened because another view of theirs was written, and blocks opened
/// because a view of theirs was mapped; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static ALIASES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Makes `block` read-only: another view of pages a copy has just taken
/// through its own view, so that a write through this one faults too. It is
/// not noted clean. The copies of title memory at this view have not taken
/// the pages: noted clean, the block would not be reported written at their
/// next check and they would keep the bytes from before the write (`guard`
/// reports a block that is read-only and not clean as written, then notes it
/// clean). A block that is clean stays so: had the pages been written since
/// its copies took them, `open` would have opened it.
fn protect_alias(g: &mut Guards, block: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    if g.protected.contains(block) {
        return;
    }
    let mut old = 0;
    // SAFETY: title view memory; only its protection changes.
    if unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) } != 0 {
        g.protected.insert(block);
        GUARD_GEN.fetch_add(1, Ordering::Release);
    }
}

/// [`protect`] for one range, without the other views of its pages.
fn protect_here(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_FREE, MEM_MAPPED, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_READONLY,
        VirtualProtect, VirtualQuery,
    };
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.insert_range(from, to);
        g.clean.insert_range(from, to);
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    let mut block = from;
    while block < to {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
        if n != 0 {
            let until = ((info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK).min(to);
            if matches!(info.State, MEM_RESERVE | MEM_FREE) {
                if block < until {
                    if info.State == MEM_FREE || info.Type != MEM_MAPPED {
                        g.holes.insert_range(block, until);
                        g.clean.insert_range(block, until);
                    } else {
                        g.absent.insert_range(block, until);
                    }
                    block = until;
                    continue;
                }
            } else if info.State == MEM_COMMIT && block < until && set(block, until) {
                note(g, block, until);
                block = until;
                continue;
            }
        }
        if set(block, block + 1) {
            note(g, block, block + 1);
        }
        block += 1;
    }
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`.
fn uncommitted_end(block: usize) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
        (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK
    } else {
        block
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded. The same physical pages at
/// every other view of them are opened too: the write that faulted (or is
/// ours, `crate::host::copy_mapped`) goes through one view, and every
/// guard over the pages must copy them again.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    let any = open_here(g, start, end);
    if start >= end || g.aliased.is_empty() {
        return any;
    }
    let (lo, hi) = (start / BLOCK, end.div_ceil(BLOCK));
    let aliased_runs = g.aliased.covered_in(lo, hi);
    if aliased_runs.is_empty() {
        return any;
    }
    g.aliased.remove_range(lo, hi);
    for (a, b) in aliased_runs {
        for block in a..b {
            for alias in crate::inplace::pages_of(block * BLOCK) {
                let ab = alias / BLOCK;
                if ab != block && open_here(g, alias, alias.saturating_add(BLOCK)) {
                    #[cfg(feature = "diagnostics")]
                    ALIASES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                g.aliased.remove_range(ab, ab.saturating_add(1));
            }
        }
    }
    any
}

/// [`open`] for one range, without the other views of its pages.
fn open_here(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    if start >= end {
        return false;
    }
    let (lo, hi) = (start / BLOCK, end.div_ceil(BLOCK));
    let runs = g.protected.covered_in(lo, hi);
    if runs.is_empty() {
        return false;
    }
    g.protected.remove_range(lo, hi);
    g.clean.remove_range(lo, hi);
    GUARD_GEN.fetch_add(1, Ordering::Release);
    for (a, b) in runs {
        let mut old = 0;
        // SAFETY: blocks we made read-only.
        let bulk = unsafe {
            VirtualProtect((a * BLOCK) as *const _, (b - a) * BLOCK, PAGE_READWRITE, &mut old)
        } != 0;
        if !bulk {
            for block in a..b {
                // SAFETY: a block we made read-only.
                unsafe {
                    VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old)
                };
            }
        }
    }
    true
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    let end = at.saturating_add(len);
    if len > 0 {
        let (lo, hi) = (at / BLOCK, end.div_ceil(BLOCK));
        if guards_read().protected.overlaps(lo, hi) {
            open(&mut guards(), at, end);
        }
    }
    // A placed texture there changes with no descriptor written.
    crate::descriptor::host::remapped(at, len);
}

/// What `VICTORY_PAGE_WRITES` asks to be done with a write the guards do
/// not see once its bytes are there ([`title_memory_written`]), as bits:
/// 1, the guarded blocks under it are opened and count as written (host
/// copies of buffers and textures read through tables take them again);
/// 2, the placed textures picked by index that lie under it are due again;
/// 4, both also for reads to plain addresses when they have finished.
/// Unset or `guards`: bit 1 alone. `0`: nothing, counted only (the default
/// until run 199). `1`: bits 1 and 2. `slots`: bit 2 alone. `all`:
/// everything.
/// (Bit 1 by default: in every run 10,000 to 50,000 guarded blocks of host
/// copies lay under such writes and kept the bytes from before; runs 197
/// and 198 opened them, 9,766 and 10,796 blocks, with nothing worse for
/// it. Bit 2 makes texture slots due and is the one that flooded the
/// queue in an earlier form: still asked for by name.)
pub(crate) fn page_writes() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_PAGE_WRITES").as_deref().map(str::trim) {
        Ok("0") => 0,
        Ok("1") => 3,
        Ok("slots") => 2,
        Ok("all") => 7,
        _ => 1,
    })
}

/// Since the start: writes into physical pages through the runtime's own
/// views that landed at addresses the title has mapped (one for each run
/// of addresses); their bytes; guarded blocks that lay under them (host
/// copies that would otherwise keep the bytes from before); reads to
/// plain addresses that finished.
pub(crate) static WRITTEN: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The bytes of a write the guards do not see are there: title memory
/// `[at, at + len)`. `pages`: DirectStorage wrote them into physical pages
/// through a view of the runtime's own, and this is an address where the
/// title has those pages mapped (no title address faulted, nothing was
/// mapped or unmapped; the shortcut this file's header names). Otherwise a
/// read to a plain address has finished, of which [`writing`] was told
/// before it began.
///
/// Upstream's c75fdba does the same for its trackers (its `Writing` marks
/// the pages once more when dropped; page destinations are reported at
/// each address showing them). Here it is counted always and acted on as
/// [`page_writes`] says. Called without the runtime's page-table lock;
/// takes the guards' lock, then, alone, the lock of the ranges kept for
/// the texture pass.
pub fn title_memory_written(at: usize, len: usize, pages: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    if len == 0 {
        return;
    }
    let mode = page_writes();
    let end = at.saturating_add(len);
    if pages {
        WRITTEN[0].fetch_add(1, Relaxed);
        WRITTEN[1].fetch_add(len as u64, Relaxed);
    } else {
        WRITTEN[3].fetch_add(1, Relaxed);
        if mode & 4 == 0 {
            return;
        }
    }
    {
        let guarded = guards_read().protected.count_in(at / BLOCK, end.div_ceil(BLOCK)) as u64;
        if pages {
            WRITTEN[2].fetch_add(guarded, Relaxed);
        }
        if mode & 1 != 0 && guarded != 0 {
            open(&mut guards(), at, end);
        }
    }
    // A placed texture there changed with no descriptor written.
    crate::descriptor::host::page_written(at, len);
}

/// The runtime mapped a view of the physical bytes `[offset, offset + len)`
/// of `section` (`crate::inplace::title_memory_view`): the view is
/// writable, so the same bytes guarded at another view must be checked
/// again — and are, the pages noted written, which also has the next check
/// guard the new view (`protect`).
pub(crate) fn views_mapped(section: usize, offset: u64, len: usize) {
    if !view_guard() {
        return;
    }
    let ranges = crate::inplace::views_of_section(section, offset, len);
    if ranges.is_empty() {
        return;
    }
    let mut g = guards();
    for &(at, n) in &ranges {
        if open_here(&mut g, at, at.saturating_add(n)) {
            #[cfg(feature = "diagnostics")]
            ALIASES[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Held shared by a read of title memory in place, and alone by the runtime
/// while it changes what title memory shows.
///
/// The runtime does not change a mapping in one step. To unmap one page of
/// a view it unmaps the whole view and maps what stays again
/// (`xmem.rs` `unmap`), and a view it maps is a reservation until the next
/// call commits it (`map`, then `commit`). For that moment pages the title
/// never gave up are "reserved, never committed", and a thread reading
/// them in place takes an access violation: run 71 in `Layout::detile`,
/// run 74 in the hash before it, both under `Staging::refresh_now`, which
/// had asked whether the texture was all mapped and been told yes. Asking
/// cannot help; not changing the mapping while such a read is under way
/// does.
static MAPPING: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping(#[allow(dead_code)] std::sync::RwLockWriteGuard<'static, ()>);

/// The runtime is about to map, unmap or move physical pages of title
/// memory: until what it returns is dropped, no read of title memory in
/// place is under way and none begins ([`title_memory_reading`]). Taken
/// around the whole change, the unmapping and the mapping again.
pub fn title_memory_remapping() -> Remapping {
    Remapping(MAPPING.write().unwrap_or_else(|e| e.into_inner()))
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(#[allow(dead_code)] std::sync::RwLockReadGuard<'static, ()>);

/// For reading title memory in place: until what it returns is dropped,
/// the runtime changes no mapping, so memory found mapped stays mapped.
/// Not to be taken twice by one thread, nor held across anything that
/// maps title memory.
pub fn title_memory_reading() -> Reading {
    Reading(MAPPING.read().unwrap_or_else(|e| e.into_inner()))
}

/// [`title_memory_reading`] if it can be had at once: `None` while the
/// runtime changes a mapping (or waits to), for a caller that has another
/// way to read and would rather not wait.
pub fn title_memory_reading_now() -> Option<Reading> {
    match MAPPING.try_read() {
        Ok(guard) => Some(Reading(guard)),
        Err(std::sync::TryLockError::Poisoned(e)) => Some(Reading(e.into_inner())),
        Err(std::sync::TryLockError::WouldBlock) => None,
    }
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded (nor known
/// unmapped).
pub fn title_memory_remapped(at: usize, len: usize) {
    // A placed texture there arrives, goes or changes with no descriptor
    // written.
    crate::descriptor::host::remapped(at, len);
    // The views the runtime told us of there are no longer what it shows.
    crate::inplace::views_gone(at, len);
    // And nothing stays in place over it.
    crate::inplace::title_memory_freeing(at, len);
    let (lo, hi) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    if !guards_read().overlaps_any(lo, hi) {
        return;
    }
    let mut g = guards();
    let changed = g.overlaps_any(lo, hi);
    g.absent.remove_range(lo, hi);
    g.holes.remove_range(lo, hi);
    g.protected.remove_range(lo, hi);
    g.clean.remove_range(lo, hi);
    g.aliased.remove_range(lo, hi);
    if changed {
        GUARD_GEN.fetch_add(1, Ordering::Release);
    }
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if open(&mut g, at, at + 1) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
        title_memory_remapped(view, 2 * BLOCK);
    }

    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_remapped(self.0.Value as usize, SIZE);
                // SAFETY: our view is still mapped.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: a new reserved pagefile section. The view holds its own
        // reference after we close the handle, including on assertion failure.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            View(view)
        };
        let start = view.0.Value as usize;
        let commit = |at: usize, len| {
            // SAFETY: these pages belong to our reserved view.
            let committed = unsafe { VirtualAlloc(at as _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, at);
        };
        let early = start + BLOCK + 3 * KERNEL_PAGE;
        let late = start + 2 * BLOCK + 3 * KERNEL_PAGE;
        commit(start, BLOCK);
        commit(early, KERNEL_PAGE);
        let refresh = || guard(start, start + SIZE).unwrap();
        assert_eq!(refresh(), [start / PAGE, early / PAGE, late / PAGE]);
        // The last block is still entirely reserved.
        assert!(!refresh().contains(&(late / PAGE)));

        commit(late, KERNEL_PAGE); // Ordinary commit, without a runtime mapping notification.
        for at in [early, late] {
            assert!(writable(at));
            // SAFETY: each address is inside its committed read-write page.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        assert_eq!(refresh(), [early / PAGE, late / PAGE]);
    }

    /// A range of view memory whose first block is not mapped is a view
    /// all the same: its blocks are reported once, then the ones written
    /// or mapped since.
    #[test]
    fn a_range_that_begins_in_an_unmapped_block_is_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of three; the middle block is mapped first.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (3 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        // Nothing mapped at all: a view with every block absent.
        assert!(is_view(p, p + 3 * BLOCK));
        let middle = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        // The range begins in the placeholder, its middle block is a view.
        assert!(is_view(p, p + 3 * BLOCK));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()), "nothing written since");
        // SAFETY: inside the mapped middle block.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        let first = map(0);
        title_memory_remapped(p, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // Memory of ours that is no view stays none.
        let mut local = [0u8; 16];
        let at = local.as_mut_ptr() as usize;
        assert!(!is_view(at, at + local.len()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// A pagefile section mapped twice: two views of the same physical
    /// pages, the second left for the test; the section handle, the two
    /// addresses.
    fn two_views(bytes: usize) -> (usize, usize, usize) {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        // SAFETY: a fresh pagefile section, mapped twice; the views keep
        // the section alive and the handle is closed by the caller.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                bytes as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            assert_ne!(one, 0);
            assert_ne!(two, 0);
            crate::inplace::title_memory_view(one, bytes, section as usize, 0);
            crate::inplace::title_memory_view(two, bytes, section as usize, 0);
            (section as usize, one, two)
        }
    }

    /// The section of [`two_views`] is closed and its views unmapped; the
    /// guard is told they are gone first.
    fn drop_views(section: usize, views: &[usize], bytes: usize) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for &view in views {
            title_memory_remapped(view, bytes);
            // SAFETY: a view of the section mapped by `two_views`.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                })
            };
        }
        // SAFETY: the section's last reference.
        unsafe { CloseHandle(section as *mut std::ffi::c_void) };
    }

    /// The title wrote the pages of a guarded texture through another view
    /// of them: the fault at that view is the guard's to see, and the
    /// guarded view's next check reports the pages written.
    #[test]
    fn a_write_through_another_view_is_reported_at_the_guarded_one() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        // The first check takes it in (and every view of its pages).
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        // Writing through the other view faults into the handler, which
        // opens that view and the guarded one: the next check sees it.
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((other + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A view of guarded pages is mapped anew: it is writable, so the
    /// guarded view checks its pages again (and guards the new view).
    #[test]
    fn a_view_mapped_beside_a_guarded_one_makes_it_check_again() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        title_memory_remapped(other, SIZE);
        crate::inplace::title_memory_view(other, SIZE, section, 0);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A write of the runtime's own, told at one view of the pages
    /// (`writing`): the other views' next checks report the pages written.
    #[test]
    fn a_write_told_at_one_view_is_reported_at_the_others() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        writing(other + BLOCK, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        // Half a page and the next: both blocks, not just the first's.
        writing(other + BLOCK / 2, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// Both views of the pages are checked, each for a copy of its own
    /// (the title shows one physical page at several addresses, and each
    /// address has its host copy): a write through either is reported at
    /// both, whichever is checked first. Guarding the pages for the first
    /// view's check must not note the second view clean, or its copy would
    /// keep the bytes from before the write.
    #[test]
    fn a_write_is_reported_at_every_checked_view() {
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0), pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0), pages(two, 1)]), "the second view's first check");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()), "nothing written yet");
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()), "nothing written yet");
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((two + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 1)]), "written through this view, checked second");
        // SAFETY: the first view, as above.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0)]));
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0)]), "written through this view, checked second");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));
        drop_views(section, &[one, two], SIZE);
    }

    /// A page shown in more than `MANY_VIEWS` views is guarded at each by
    /// itself: the others stay writable, and nothing looks through them all.
    #[test]
    fn a_page_of_many_views_is_guarded_one_view_at_a_time() {
        let (section, one, two) = two_views(BLOCK);
        let filler = one + (1 << 45);
        for n in 0..MANY_VIEWS {
            crate::inplace::title_memory_view(filler + n * BLOCK, BLOCK, section, 0);
        }
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        assert!(writable(two), "another view of it is left alone");
        // SAFETY: the guarded view, opened by the handler.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        title_memory_remapped(filler, MANY_VIEWS * BLOCK);
        drop_views(section, &[one, two], BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }

    #[test]
    fn range_set_merges_splits_and_queries_gaps() {
        let mut s = RangeSet::new();
        assert!(s.is_empty());
        s.insert_range(10, 20);
        s.insert_range(20, 30);
        s.insert_range(5, 10);
        assert!(s.contains_range(5, 30));
        assert_eq!(s.count_in(0, 40), 25);
        assert_eq!(s.covered_end(5), 30);
        assert_eq!(s.uncovered_in(5, 30), Vec::<(usize, usize)>::new());

        // Split the middle out.
        s.remove_range(12, 18);
        assert!(!s.contains_range(5, 30));
        assert!(s.contains_range(5, 12));
        assert!(s.contains_range(18, 30));
        assert_eq!(s.uncovered_in(0, 35), vec![(0, 5), (12, 18), (30, 35)]);
        assert_eq!(s.covered_in(8, 22), vec![(8, 12), (18, 22)]);
        assert_eq!(s.count_in(8, 22), 8);

        // Remove across multiple intervals.
        s.remove_range(10, 25);
        assert_eq!(s.covered_in(0, 40), vec![(5, 10), (25, 30)]);
        assert_eq!(s.count_in(0, 40), 10);
        assert!(!s.overlaps(0, 5));
        assert!(s.overlaps(0, 6));
        assert!(s.overlaps(9, 15));
        assert!(!s.overlaps(10, 25));
        assert!(s.overlaps(24, 26));
        assert!(!s.overlaps(30, 40));
    }

    #[test]
    fn placeholder_holes_join_clean_until_remapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh 3-block placeholder and 2-block pagefile section.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| unsafe {
            VirtualFree(
                (p + block * BLOCK) as *mut _,
                BLOCK,
                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
            );
            let view = MapViewOfFile3(
                section,
                GetCurrentProcess(),
                (p + block * BLOCK) as *const _,
                (block * BLOCK) as u64,
                BLOCK,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            );
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let b0 = p / BLOCK;
        assert!(guard(p, p + 3 * BLOCK).is_some());
        {
            let g = guards();
            assert!(g.holes.contains_range(b0 + 1, b0 + 3));
            assert!(g.clean.contains_range(b0, b0 + 3));
            assert!(!g.absent.contains(b0 + 1));
        }
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        {
            let g = guards();
            assert!(!g.holes.contains(b0 + 1));
            assert!(g.holes.contains(b0 + 2));
            assert!(!g.clean.contains(b0 + 1));
        }
        assert_eq!(guard(p, p + 3 * BLOCK), Some(vec![(p + BLOCK) / PAGE]));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        assert!(guards().clean.contains_range(b0, b0 + 3));
        // SAFETY: our views, placeholder, and section.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::{title_memory_reading, title_memory_remapping};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// A change of mapping waits for a read in place that is under way, and
    /// a read in place waits for a change of mapping.
    #[test]
    fn reads_in_place_and_remappings_wait_for_each_other() {
        // A reader holds; the change does not begin until it lets go.
        let changed = Arc::new(AtomicBool::new(false));
        let reading = title_memory_reading();
        let (began, wait) = mpsc::channel();
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _changing = title_memory_remapping();
                changed.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!changed.load(Ordering::SeqCst), "the mapping changed under a read in place");
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
        // A change holds; the read does not begin until it is done.
        let read = Arc::new(AtomicBool::new(false));
        let changing = title_memory_remapping();
        let (began, wait) = mpsc::channel();
        let reader = {
            let read = read.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _reading = title_memory_reading();
                read.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!read.load(Ordering::SeqCst), "a read in place began during a change of mapping");
        drop(changing);
        reader.join().unwrap();
        assert!(read.load(Ordering::SeqCst));
    }
}
