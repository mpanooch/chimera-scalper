// include/shared_book.h
#pragma once
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <string>
#include <cstdio>

#pragma pack(push, 1)

static constexpr int CHI_MAX_DEPTH = 50;  // match your Bybit subscription
static constexpr int CHI_SYM_LEN   = 16;

struct ChiBookHeader {
    uint64_t ts_ns;                 // last update (ns)
    char     symbol[CHI_SYM_LEN];   // null-terminated
    uint32_t depth;                 // <= CHI_MAX_DEPTH
};

struct ChiLevel {
    double price; // raw double (no tick scaling)
    double qty;   // raw double
};

struct ChiSharedBook {
    ChiBookHeader hdr;
    ChiLevel bids[CHI_MAX_DEPTH];   // sorted: price DESC
    ChiLevel asks[CHI_MAX_DEPTH];   // sorted: price ASC
};

#pragma pack(pop)

// Compile-time guardrails (catch ABI drift at build time)
static_assert(offsetof(ChiSharedBook, bids) > 0, "bids offset must be > 0");
static_assert(offsetof(ChiSharedBook, asks) > offsetof(ChiSharedBook, bids), "asks must follow bids");
static_assert(sizeof(ChiLevel) == sizeof(double)*2, "ChiLevel must be packed (price, qty)");

// ---------------- Writer helper ----------------
inline void write_shared_book(ChiSharedBook* out,
                              const std::string& symbol,
                              const ChiLevel* topBids, size_t nbids,
                              const ChiLevel* topAsks, size_t nasks,
                              uint64_t ts_ns) {
    out->hdr.ts_ns = ts_ns;
    std::snprintf(out->hdr.symbol, CHI_SYM_LEN, "%s", symbol.c_str());
    const uint32_t d = static_cast<uint32_t>(
        std::min({ (size_t)CHI_MAX_DEPTH, nbids, nasks })
    );
    out->hdr.depth = d;

    for (uint32_t i = 0; i < d; ++i) out->bids[i] = topBids[i];
    for (uint32_t i = 0; i < d; ++i) out->asks[i] = topAsks[i];

#ifdef CHIMERA_DEBUG_BOOK
    if (d > 0) {
        std::fprintf(stderr,
            "[FEED] %s ts=%llu | bid0=%.8f x %.3f | ask0=%.8f x %.3f | depth=%u\n",
            out->hdr.symbol,
            (unsigned long long)out->hdr.ts_ns,
            out->bids[0].price, out->bids[0].qty,
            out->asks[0].price, out->asks[0].qty,
            out->hdr.depth
        );
    }
#endif
}
