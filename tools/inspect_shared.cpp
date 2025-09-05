// tools/inspect_shared.cpp
#include "shared_book.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    const char* path = (argc > 1 ? argv[1] : "/tmp/chimera_orderbook.dat");
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }

    struct stat st{};
    if (fstat(fd, &st) != 0) { perror("fstat"); return 1; }
    size_t len = st.st_size;

    void* p = mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); return 1; }

    auto* book = reinterpret_cast<const ChiSharedBook*>(p);

    std::printf("file=%s size=%zu\n", path, len);
    std::printf("sizeof(ChiSharedBook)=%zu off(bids)=%zu off(asks)=%zu\n",
        sizeof(ChiSharedBook), offsetof(ChiSharedBook, bids), offsetof(ChiSharedBook, asks));

    std::printf("symbol=%s ts_ns=%llu depth=%u\n",
        book->hdr.symbol, (unsigned long long)book->hdr.ts_ns, book->hdr.depth);

    if (book->hdr.depth > 0) {
        std::printf("bid0=%.10f x %.6f | ask0=%.10f x %.6f\n",
            book->bids[0].price, book->bids[0].qty,
            book->asks[0].price, book->asks[0].qty);
    }
    munmap(p, len);
    close(fd);
    return 0;
}
