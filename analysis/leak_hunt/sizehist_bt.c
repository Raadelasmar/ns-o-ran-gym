/* sizehist.c -- LD_PRELOAD live-allocation size histogram.
 *
 * Answers one question: when the heap grows, what SIZE are the live chunks?
 * That names the object class (~1.4 kB => ns-3 Packet buffers, ~64 B =>
 * std::map nodes, ~100 B => event nodes) without any backtrace machinery,
 * which is why it is cheap enough to run for 50 steps.
 *
 * 16-byte header per allocation preserves malloc's 16-byte alignment.  The
 * header holds the request size plus a magic, so free() finds the right
 * bucket and passes through any pointer that did not come from us.
 *
 * Buckets are power-of-two by request size, 0..31.  Counts are live
 * (incremented on alloc, decremented on free), so a dump is a live-set
 * profile, not an allocation trace.
 *
 * Reentrancy: a thread-local flag guards the dlsym bootstrap.  dlsym itself
 * allocates, so allocations made before real_malloc resolves come from a
 * static bump arena that free() recognises by address range and ignores.
 *
 * Expected slowdown: ~5-10%.  No extra thread is created, so this adds no
 * second glibc arena (unlike mallocinfo_sampler.c, which showed up cleanly
 * as heap nr=1 at ~132 KB).
 *
 * Build:
 *   gcc -O2 -fPIC -shared -o sizehist.so sizehist.c -ldl
 *
 * Env:
 *   SIZEHIST_OUT       output path (default /tmp/sizehist.csv)
 *   SIZEHIST_INTERVAL  dump every N allocations (default 2000000)
 *
 * Output CSV: seq,bucket,live_count,live_bytes  (one block per dump)
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <execinfo.h>

#define MAGIC 0x5A1E4157U
#define NBUCK 32
#define HDR   16

static void *(*real_malloc)(size_t);
static void  (*real_free)(void *);
static void *(*real_realloc)(void *, size_t);

static __thread int in_hook;

/* bump arena for allocations made during the dlsym bootstrap */
static char boot_arena[65536];
static size_t boot_used;

static volatile long live_count[NBUCK];
static volatile long live_bytes[NBUCK];
static volatile long alloc_seq;
static long dump_interval = 2000000;
static const char *out_path;

/* ---- scoped backtrace attribution -------------------------------------
 * Backtraces are captured ONLY for allocations whose bucket equals
 * SIZEHIST_TRACE_BUCKET, so the cost is paid on a small fraction of calls.
 * The stack hash lives in the spare uint32 of the 16-byte header, so free()
 * decrements the right row without any pointer-keyed lookup table. */
#define NSTK 12
#define NTAB 4096
struct stk { uint32_t hash; long live_bytes; long live_cnt; int depth; void *fr[NSTK]; };
static struct stk tab[NTAB];
static int trace_bucket = -1;

static struct stk *row_for(uint32_t h)
{
    uint32_t i = h % NTAB;
    for (uint32_t n = 0; n < NTAB; n++, i = (i + 1) % NTAB) {
        if (tab[i].hash == h) return &tab[i];
        if (tab[i].hash == 0) { tab[i].hash = h; return &tab[i]; }
    }
    return NULL;
}

static int bucket_of(size_t n)
{
    int b = 0;
    while ((size_t)1 << b < n && b < NBUCK - 1) b++;
    return b;
}

static void init_real(void)
{
    if (real_malloc) return;
    in_hook = 1;
    real_malloc  = dlsym(RTLD_NEXT, "malloc");
    real_free    = dlsym(RTLD_NEXT, "free");
    real_realloc = dlsym(RTLD_NEXT, "realloc");
    in_hook = 0;
    out_path = getenv("SIZEHIST_OUT");
    if (!out_path) out_path = "/tmp/sizehist.csv";
    const char *iv = getenv("SIZEHIST_INTERVAL");
    if (iv) dump_interval = atol(iv);
    const char *tb = getenv("SIZEHIST_TRACE_BUCKET");
    if (tb) trace_bucket = atoi(tb);
}

static void dump(void)
{
    FILE *f = fopen(out_path, "a");
    if (!f) return;
    fprintf(f, "# seq=%ld pid=%d\n", alloc_seq, getpid());
    for (int b = 0; b < NBUCK; b++)
        if (live_count[b])
            fprintf(f, "%ld,%d,%ld,%ld\n", alloc_seq, b, live_count[b], live_bytes[b]);
    if (trace_bucket >= 0) {
        for (int i = 0; i < NTAB; i++) {
            if (!tab[i].hash || tab[i].live_bytes <= 0) continue;
            fprintf(f, "STK,%u,%ld,%ld", tab[i].hash, tab[i].live_cnt, tab[i].live_bytes);
            char **sym = backtrace_symbols(tab[i].fr, tab[i].depth);
            for (int k = 0; k < tab[i].depth; k++)
                fprintf(f, "|%s", sym ? sym[k] : "?");
            fprintf(f, "\n");
            if (sym) real_free(sym);
        }
    }
    fclose(f);
}

static void *tag(void *raw, size_t n)
{
    if (!raw) return NULL;
    uint32_t *h = (uint32_t *)raw;
    h[0] = MAGIC;
    h[1] = 0;
    *(size_t *)(h + 2) = n;
    int b = bucket_of(n);
    __sync_fetch_and_add(&live_count[b], 1);
    __sync_fetch_and_add(&live_bytes[b], (long)n);
    if (b == trace_bucket && !in_hook) {
        in_hook = 1;                       /* backtrace() itself mallocs */
        void *fr[NSTK];
        int d = backtrace(fr, NSTK);
        uint32_t hh = 2166136261u;
        for (int k = 2; k < d; k++) {      /* skip tag()/malloc frames */
            uintptr_t v = (uintptr_t)fr[k];
            for (int q = 0; q < 8; q++) { hh ^= (uint32_t)((v >> (q * 8)) & 0xff); hh *= 16777619u; }
        }
        if (!hh) hh = 1;
        struct stk *r = row_for(hh);
        if (r) {
            if (!r->depth) { r->depth = d < NSTK ? d : NSTK;
                             for (int k = 0; k < r->depth; k++) r->fr[k] = fr[k]; }
            __sync_fetch_and_add(&r->live_bytes, (long)n);
            __sync_fetch_and_add(&r->live_cnt, 1);
            h[1] = hh;
        }
        in_hook = 0;
    }
    long s = __sync_fetch_and_add(&alloc_seq, 1);
    if (dump_interval > 0 && s > 0 && s % dump_interval == 0) {
        if (!in_hook) { in_hook = 1; dump(); in_hook = 0; }
    }
    return (char *)raw + HDR;
}

void *malloc(size_t n)
{
    if (!real_malloc) {
        init_real();
        if (!real_malloc) {            /* still bootstrapping: bump arena */
            size_t need = (n + 15) & ~(size_t)15;
            if (boot_used + need > sizeof boot_arena) return NULL;
            void *p = boot_arena + boot_used;
            boot_used += need;
            return p;
        }
    }
    if (in_hook) return real_malloc(n);
    return tag(real_malloc(n + HDR), n);
}

void *calloc(size_t nm, size_t sz)
{
    size_t n = nm * sz;
    void *p = malloc(n);
    if (p) memset(p, 0, n);
    return p;
}

static int is_boot(void *p)
{
    return (char *)p >= boot_arena && (char *)p < boot_arena + sizeof boot_arena;
}

void free(void *p)
{
    if (!p) return;
    if (is_boot(p)) return;
    if (!real_free) { init_real(); if (!real_free) return; }
    if (in_hook) { real_free(p); return; }
    uint32_t *h = (uint32_t *)((char *)p - HDR);
    if (h[0] != MAGIC) { real_free(p); return; }   /* not ours */
    size_t n = *(size_t *)(h + 2);
    int b = bucket_of(n);
    __sync_fetch_and_sub(&live_count[b], 1);
    __sync_fetch_and_sub(&live_bytes[b], (long)n);
    if (h[1]) {
        struct stk *r = row_for(h[1]);
        if (r) { __sync_fetch_and_sub(&r->live_bytes, (long)n);
                 __sync_fetch_and_sub(&r->live_cnt, 1); }
        h[1] = 0;
    }
    h[0] = 0;
    real_free(h);
}

void *realloc(void *p, size_t n)
{
    if (!p) return malloc(n);
    if (is_boot(p)) { void *q = malloc(n); if (q) memcpy(q, p, n); return q; }
    if (!real_realloc) { init_real(); if (!real_realloc) return NULL; }
    uint32_t *h = (uint32_t *)((char *)p - HDR);
    if (h[0] != MAGIC) return real_realloc(p, n);
    size_t old = *(size_t *)(h + 2);
    int ob = bucket_of(old);
    __sync_fetch_and_sub(&live_count[ob], 1);
    __sync_fetch_and_sub(&live_bytes[ob], (long)old);
    return tag(real_realloc(h, n + HDR), n);
}

int posix_memalign(void **out, size_t align, size_t n)
{
    if (align <= HDR) {            /* our header already gives 16B alignment */
        void *p = malloc(n);
        if (!p) return 12;
        *out = p;
        return 0;
    }
    /* Over-aligned: hand straight to glibc, untracked.  Returning a
     * mis-aligned pointer would corrupt ns-3 silently, which is worse
     * than losing these few allocations from the histogram. */
    static int (*real_pm)(void **, size_t, size_t);
    if (!real_pm) { in_hook = 1; real_pm = dlsym(RTLD_NEXT, "posix_memalign"); in_hook = 0; }
    return real_pm(out, align, n);
}

__attribute__((destructor)) static void fini(void) { in_hook = 1; dump(); }
