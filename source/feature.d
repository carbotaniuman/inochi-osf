import remedian;

import std.math : isNaN;

class FeatureExtractor {

}

class Feature {
    private Remedian remedian;
    private float min = float.nan;
    private float max = float.nan;
    private float hardMin = float.nan;
    private float hardMax = float.nan;

    // TODO: I have no idea what `maxUpdates`
    // is supposed to be here - seems like it's
    // some kind of time-based limiter
    private int maxUpdates;
    private float threshold, alpha, hardFactor, decay;

    private float last = 0.0;
    private float currentMedian = 0.0;
    private int updateCount = 0;
    private int firstSeen = -1;
    private bool updating = true;

    this(int maxUpdates = 0, float threshold = 0.15, float alpha = 0.2, float hardFactor = 0.15, float decay = 0.001) {
        this.remedian = new Remedian();
        this.maxUpdates = maxUpdates;

        this.threshold = threshold;
        this.alpha = alpha;
        this.hardFactor = hardFactor;
        this.decay = decay;
    }

    float update(float x, int now = 0) {
        if (this.maxUpdates > 0) {
            if (this.firstSeen == -1) {
                this.firstSeen = now;
            }
        }
        auto next = this.updateState(x, now);
        auto filtered = this.last * this.alpha + next * (1.0 - this.alpha);
        this.last = filtered;
        return filtered;
    }

    // TODO: I have no idea if any of this works as I
    // was unable to meaningfully cross-check it
    private float updateState(float x, int now) {
        auto shouldUpdate = this.updating && (this.maxUpdates == 0 || (now - this.firstSeen) < this.maxUpdates);
        if (shouldUpdate) {
            this.remedian.push(x);
            this.currentMedian = this.remedian.median();
        } else {
            this.updating = false;
        }

        auto median = this.currentMedian;
        if (this.min.isNaN) {
            auto t = (median - x) / median;
            if (x < median && t > this.threshold) {
                if (shouldUpdate) {
                    this.min = x;
                    this.hardMin = this.min + this.hardFactor * (median - this.min);
                }
                return -1;
            }
            return 0;
        } else {
            if (x < this.min) {
                if (shouldUpdate) {
                    this.min = x;
                    this.hardMin = this.min + this.hardFactor * (median - this.min);
                }
                return -1;
            }
        }

        if (this.max.isNaN) {
            auto t = (x - median) / median;
            if (x > median && t > this.threshold) {
                if (shouldUpdate) {
                    this.max = x;
                    this.hardMax = this.max - this.hardFactor * (this.max - median);
                }
                return 1;
            }
            return 0;
        } else {
            if (x > this.max) {
                if (shouldUpdate) {
                    this.max = x;
                    this.hardMax = this.max - this.hardFactor * (this.max - median);
                }
                return 1;
            }
        }

        if (shouldUpdate) {
            if (this.min < this.hardMin) {
                this.min = this.hardMin * this.decay + this.min * (1.0 - this.decay);
            }

            if (this.max > this.hardMax) {
                this.max = this.hardMax * this.decay + this.max * (1.0 - this.decay);
            }
        }

        if (x < median) {
            return -(1.0 - (x - this.min) / (median - this.min));
        } else if (x > median) {
            return (x - median) / (this.max - median);
        }

        return 0;
    }
}