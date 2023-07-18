module remedian;

import std.algorithm : sort;
import std.array : Appender;
import std.math : isNaN;

private float rawMedian(float[] arr) {
    size_t n = arr.length;
    size_t p = n / 2;
    size_t q = n / 2;

    if (n < 3) {
        p = 0;
        q = n - 1;

        if (p == q) {
            return arr[p];
        } else {
            return (arr[p] + arr[q]) / 2.0;
        }
    } else {
        auto sorted = arr.dup.sort;

        // For even-length lists, use mean of mid 2 nums
        if (n % 2 == 0) {
            q = p - 1;
        }

        if (p == q) {
            return sorted[p];
        } else {
            return (sorted[p] + sorted[q]) / 2.0;
        }
    }
}

class Remedian {
    private size_t k;
    private Appender!(float[]) all;
    private Remedian more = null;

    // If the actual median value is NaN,
    // this will cause excess computations,
    // but that's probably fine.
    private float cachedMedian = float.nan;

    this(size_t k = 64) {
        this.k = k;
        this.all.reserve(k);
    }

    void push(float x) {
        this.cachedMedian = float.nan;
        this.all.put(x);

        if (this.all[].length == this.k) {
            if (this.more is null) {
                this.more = new Remedian(this.k);
            }
            this.more.push(rawMedian(this.all[]));
            this.all.clear();
        }
    }

    float median() {
        if (this.more) {
            return this.more.median;
        } else {
            if (this.cachedMedian.isNaN) {
                this.cachedMedian = rawMedian(this.all[]);
            }
            return this.cachedMedian;
        }
    }
}