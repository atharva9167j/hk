/**
 * HK C/C++ Integration Example
 * Demonstrates loading an HK file, checking 128-byte alignment,
 * reading tensors and appendix evolutionary records, and dequantizing.
 */

#include <stdio.h>
#include <stdlib.h>
#include "../../include/hk.h"

int main(int argc, char** argv) {
    const char* file_path = (argc > 1) ? argv[1] : "models/hwr_sparse_2_4.hk";
    printf("Loading HK model via C ABI: %s\n", file_path);

    hk_reader_t* reader = hk_open(file_path);
    if (!reader) {
        printf("Failed to open HK file: %s\n", file_path);
        return 1;
    }

    uint64_t tensor_count = hk_get_tensor_count(reader);
    printf("Successfully opened container. Total tensors: %llu\n", (unsigned long long)tensor_count);

    for (uint64_t i = 0; i < tensor_count; i++) {
        hk_tensor_info_t info;
        if (hk_get_tensor_info(reader, i, &info) == 0) {
            int is_aligned = (info.data_offset % 128 == 0) ? 1 : 0;
            printf("  [%llu] %-35s | Storage: 0x%02X | Size: %6llu B | 128-byte Aligned: %s\n",
                (unsigned long long)i,
                info.name,
                info.storage_type,
                (unsigned long long)info.data_size,
                is_aligned ? "YES" : "NO"
            );
        }
    }

    // Inspect Appendix Region
    uint64_t app_count = hk_appendix_get_count(reader);
    printf("\nAppendix evolutionary records: %llu\n", (unsigned long long)app_count);
    for (uint64_t i = 0; i < app_count; i++) {
        hk_appendix_entry_t entry;
        if (hk_appendix_get_entry(reader, i, &entry) == 0) {
            printf("  Appendix [%llu] Gen %u: %s (Type: 0x%02X, Acc: %.2f%%, Pass: %.2f%%)\n",
                (unsigned long long)i,
                entry.generation,
                entry.name,
                entry.entry_type,
                entry.metric_acc * 100.0f,
                entry.metric_pass * 100.0f
            );
        }
    }

    hk_close(reader);
    printf("Closed HK container cleanly.\n");
    return 0;
}
