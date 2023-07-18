module faceinfo;

import inmath;
import retinaface : filterIndices;

import mir.ndslice;
import tracker : ModelKind, norm, ResultData, EyeState, Tracker;
import face3d;

import kaleidic.lubeck : inv, mtimes;

import std.algorithm : setIntersection;
import std.stdio;

struct Point {
    float x, y;

    Point opBinary(string op)(Point r) const if((op == "+") || (op == "-")) {
        Point ret;
        ret.x = mixin("this.x" ~ op ~ "r.x");
        ret.y = mixin("this.y" ~ op ~ "r.y");
        return ret;
    }

    bool isFinite() const {
        import std.math;

        if(isNaN(x) || isInfinity(x)) {
            return false;
        }

        if(isNaN(y) || isInfinity(y)) {
            return false;
        }
        return true;
    }
}

struct FaceData {
    float x, y, width, height;
}

static const double updateCountDelta = 75.0;
static const double updateCountMax = 75.0;

// Honestly I have no idea what this class does and
// it seems to do way too many things, but alas
class FaceInfo {
private:
        Tracker tracker;
public:
        Slice!(size_t*, 1, Contiguous) contourPoints;
        bool alive = true;
        int frameCount = 0;
        int failureCount = 0;
        float conf = float.nan;
        float pnpError = float.nan;
        Point coord = Point(float.nan, float.nan);

        Slice!(float*, 2, Contiguous) lms = slice!float(0, 0);
        Slice!(float*, 2, Contiguous) updateCounts;
        Slice!(float*, 2, Contiguous) face3d;
        Slice!(float*, 2, Contiguous) points3d;
        Slice!(float*, 2, Contiguous) contour;

        EyeState[] eyeState = [];

        quat rotation = quat(float.nan, float.nan, float.nan, float.nan);
        vec3 translation = vec3(float.nan, float.nan, float.nan);

    this(Tracker tracker, ModelKind type) {
        this.tracker = tracker;

        this.face3d = defaultFace3d.fuse.dup;
        this.updateCounts = slice!float([66, 2], 0);

        if (type == ModelKind.T) {
            this.contourPoints = [0, 2, 8, 14, 16, 27, 30, 33].sliced!size_t;
        } else {
            this.contourPoints = [0, 1, 8, 15, 16, 27, 28, 29, 30, 31, 32, 33, 34, 35].sliced!size_t;
        }

        updateContour();
    }

    void resetFace3d() {
        this.face3d = defaultFace3d.fuse.dup;
    }

    private void reset() {
        this.failureCount = 0;
        this.conf = float.nan;
        this.pnpError = float.nan;

        this.lms = slice!float(0, 0);
        this.points3d = typeof(this.points3d).init;

        this.eyeState = [];
        
        // self.success = None
        // self.eye_blink = None
        // self.bbox = None
        // if self.tracker.max_feature_updates < 1:
        //     self.features = FeatureExtractor(0)
        // self.current_features = {}
        
        this.updateCounts = slice!float([66, 2], 0);
        this.updateContour();

        this.rotation = quat(float.nan, float.nan, float.nan, float.nan);
        this.translation = vec3(float.nan, float.nan, float.nan);
    }

    final void updateContour() {
        this.contour = this.face3d[this.contourPoints].fuse;
    }

    void update(int frameCount) {
        this.frameCount = frameCount;
        this.reset();
    }

    void update(int frameCount, ResultData data, Point coord) {
        this.frameCount = frameCount;
        this.alive = true;

        this.conf = data.conf - data.bonus;
        this.lms = data.lms;
        this.eyeState = data.eyeState;
        this.coord = coord;
    }

    void adjust3d(Slice!(float*, 2, Contiguous) points3d, ModelKind kind, bool staticModel) {
        if (this.conf < 0.4 || this.pnpError > 300) {
            return;
        }
        float[3] euler = [
            this.rotation.roll / PI * 180.0,
            this.rotation.pitch / PI * 180.0,
            this.rotation.yaw / PI * 180.0
        ];

        if (kind != ModelKind.T && !staticModel) {

            auto changedAny = false;
            auto updateType = -1;
            Slice!(float*, 2, Contiguous) updated;
            
            // just a block so break works
            do {
                import mir.random.algorithm : randomSlice;
                import mir.random.variable: uniformVar;
                auto sample = uniformVar(-0.01, 0.01).randomSlice([66, 3]);
                sample[] += 1.0;
                sample[30, 0..$] = 1.0;

                if (euler[0] > -165.0 && euler[0] < 145.0) {
                    continue;
                } else if (euler[1] > -10.0 && euler[1] < 20.0) {
                    sample[0..$, 2] = 1.0;
                    updateType = 0;
                } else {
                    sample[0..$, 0..2] = 1.0;
                    if (euler[2] > 120.0 || euler[2] < 60.0) {
                        continue;
                    } else if (euler[1] < -10.0) {
                        updateType = 1;
                        foreach (i; [0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 19, 20, 21, 31, 32, 36, 37, 38, 39, 40, 41, 48, 49, 56, 57,
                           58, 59, 65]) {
                            sample[i, 2] = 1.0;
                        }
                    } else {
                        updateType = 1;
                        foreach (i; [9, 10, 11, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26, 34, 35, 42, 43, 44, 45, 46, 47, 51, 52,
                           53, 54, 61, 62, 63]) {
                            sample[i, 2] = 1.0;
                        }
                    }
                }

                auto eq = zip(this.updateCounts[0..$, updateType], this.updateCounts[0..$, abs(updateType - 1)])
                    .map!((a, b) => a < b + updateCountDelta);
                // eligible is just always set, so ignore all the codepaths above setting it
                auto eligible = filterIndices(eq);
                if (eligible.length == 0) {
                    break;
                }

                import project;
                auto rvec = this.rotation.toAxisAngle().vector.sliced;
                auto tvec = this.translation.vector.sliced;
                auto projectOutput = slice!double(eligible.length, 2);

                updated = this.face3d[0..66].dup;
                auto oProjected = slice!float([66, 2], 1);
                {
                    auto indexed = this.face3d.indexed(eligible).fuse;
                    projectPoints(indexed, rvec, tvec, this.tracker.camera, slice!float(0), projectOutput);

                    foreach (i, v; eligible) {
                        oProjected[v, 0..$] = projectOutput[i];
                    }
                }

                auto c = updated * sample;
                auto cProjected = slice!float([66, 2], 1);
                {
                    auto indexed = c.indexed(eligible).fuse;
                    projectPoints(indexed, rvec, tvec, this.tracker.camera, slice!float(0), projectOutput);

                    foreach (i, v; eligible) {
                        cProjected[v, 0..$] = projectOutput[i];
                    }
                }

                auto dO = slice!float([66], 1);
                {
                    auto subbed = oProjected.indexed(eligible) - this.lms[0..$, 0..2].indexed(eligible);
                    auto normed = subbed.byDim!0.map!((a) => norm(a[0], a[1]));
                    foreach (i, v; eligible) {
                        dO[v] = normed[i];
                    }
                }

                auto dC = slice!float([66], 1);
                {
                    auto subbed = cProjected.indexed(eligible) - this.lms[0..$, 0..2].indexed(eligible);
                    auto normed = subbed.byDim!0.map!((a) => norm(a[0], a[1]));
                    foreach (i, v; eligible) {
                        dC[v] = normed[i];
                    }
                }

                auto dOdCEq = zip(dC, dO).map!((a, b) => a < b);
                auto indices = filterIndices(dOdCEq);

                if (indices.length > 0) {
                    import std.array : array;
                    auto intersected = setIntersection(indices, eligible).array;
                    this.updateCounts[0..$, updateType].indexed(intersected)[] += 1;

                    foreach (v; eligible) {
                        updated[v, 0..$] = c[v];
                        oProjected[v, 0..$] = cProjected[v];
                    }
                    changedAny = true;
                } else {
                    break;
                }
            } while (false);

            if (changedAny) {
                auto weights = slice!float([66, 3], 0);
                foreach (i; 0..66) {
                    auto value = this.lms[i, 2];
                    if (value > 0.7) {
                        value = 1.0;
                    }
                    weights[i, 0] = 1.0 - value;
                    weights[i, 1] = 1.0 - value;
                    weights[i, 2] = 1.0 - value;
                }

                auto updateIndices = filterIndices(this.updateCounts[0..$, updateType].map!((a) => a < updateCountMax));

                foreach (i; updateIndices) {
                    this.face3d[i, 0..$] = this.face3d[i] * weights[i]
                        + updated[i] * (1.0 - weights[i]);
                }
                this.updateContour();
                this.points3d = normalizePoints3d(this.points3d);
            }

            // TODO: calculate stuff for features
        }
    }

    private Slice!(float*, 2, Contiguous) normalizePoints3d(Slice!(float*, 2, Contiguous) points3d) {
        // TODO: actually normalize
        return points3d;
    }

        // def normalize_pts3d(self, pts_3d):
        // # Calculate angle using nose
        // pts_3d[:, 0:2] -= pts_3d[30, 0:2]
        // alpha = angle(pts_3d[30, 0:2], pts_3d[27, 0:2])
        // alpha -= np.deg2rad(90)

        // R = np.matrix([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        // pts_3d[:, 0:2] = (pts_3d - pts_3d[30])[:, 0:2].dot(R) + pts_3d[30, 0:2]

        // # Vertical scale
        // pts_3d[:, 1] /= np.mean((pts_3d[27:30, 1] - pts_3d[28:31, 1]) / self.base_scale_v)

        // # Horizontal scale
        // pts_3d[:, 0] /= np.mean(np.abs(pts_3d[[0, 36, 42], 0] - pts_3d[[16, 39, 45], 0]) / self.base_scale_h)

        // return pts_3d
}
