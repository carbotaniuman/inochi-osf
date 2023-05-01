module faceinfo;

import mir.ndslice;
import tracker : ModelKind, ResultData, EyeState, Tracker;

struct Point {
    float x, y;

    Point opBinary(string op)(Point r) const if((op == "+") || (op == "-")) {
        Point ret;
        ret.x = mixin("this.x" ~ op ~ "r.x");
        ret.y = mixin("this.y" ~ op ~ "r.y");
        return ret;
    }
}

struct FaceData {
    float x, y, width, height;
}

// Honestly I have no idea what this class does and
// it seems to do way too many things, but alas
class FaceInfo {
private:
        Tracker tracker;
public:
        Slice!(size_t*, 1, Contiguous) contourPoints;
        bool alive = false;
        int frameCount = 0;
        float conf = -1;
        Point coord = Point(float.nan, float.nan);
        Slice!(float*, 2, Contiguous) lms = slice!float(0, 0);
        EyeState[] eyeState = [];

    this(Tracker tracker, ModelKind type) {
        this.tracker = tracker;

        if (type == ModelKind.T) {
            this.contourPoints = [0, 2, 8, 14, 16, 27, 30, 33].sliced!size_t;
        } else {
            this.contourPoints = [0, 1, 8, 15, 16, 27, 28, 29, 30, 31, 32, 33, 34, 35].sliced!size_t;
        }
    }

    void update(int frameCount) {
        this.frameCount = frameCount;
        this.alive = false;
    }

    void update(int frameCount, ResultData data, Point coord) {
        this.frameCount = frameCount;
        this.alive = true;

        this.conf = data.conf - data.bonus;
        this.lms = data.lms;
        this.eyeState = data.eyeState;
        this.coord = coord;
    }
}
