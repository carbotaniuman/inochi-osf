module retinaface;

import std.algorithm : fold, min, max;
import std.array : array, staticArray;
import std.conv;
import std.stdio;
import std.string : toStringz;
import std.typecons : tuple;

import bindbc.onnxruntime.config;
import bindbc.onnxruntime.v12.bind;
import bindbc.onnxruntime.v12.types;

import mir.ndslice.allocation: slice;
import mir.ndslice.concatenation : concatenation;
import mir.ndslice.dynamic;
import mir.ndslice.slice;
import mir.ndslice.topology;
import dcv.core;
import dcv.imgproc;

import face;
import ortdata;

import fghj;

immutable priorBoxJson = import("priorbox_640x640.json");

import std.traits : allSameType;
I[] nms(I = size_t, BoxIterator, SliceKind boxKind, ScoreIterator, SliceKind scoreKind, T)
    (Slice!(BoxIterator, 2, boxKind) boxes, Slice!(ScoreIterator, 1, scoreKind) scores, T thresh)
    if (allSameType!(boxes.DeepElement, scores.DeepElement, T)) {
    import mir.ndslice.sorting : makeIndex;

    auto x1 = boxes[0..$, 0];
    auto y1 = boxes[0..$, 1];
    auto x2 = boxes[0..$, 2];
    auto y2 = boxes[0..$, 3];

    auto areas = (x2 - x1 + 1) * (y2 - y1 + 1);
    auto order = scores.makeIndex!I.reversed.slice;

    I[] ret = [];
    while (order.length > 0) {
        auto i = order[0];
        ret ~= i;

        auto xx1 = x1[order[1..$]].map!((a) => max(a, x1[i]));
        auto yy1 = y1[order[1..$]].map!((a) => max(a, y1[i]));
        auto xx2 = x2[order[1..$]].map!((a) => max(a, x2[i]));
        auto yy2 = y2[order[1..$]].map!((a) => max(a, y2[i]));

        auto w = (xx2 - xx1 + 1).map!((a) => max(a, 0.0));
        auto h = (yy2 - yy1 + 1).map!((a) => max(a, 0.0));

        auto inter = w * h;
        auto ovr = inter / (areas[i] + areas[order[1..$]] - inter);
    
        auto inds = filterIndices!(a => a <= thresh)(ovr);
        inds[] += 1;

        order = order.indexed(inds).slice;
    }

    return ret;
}

class RetinaFace {
private:
    Slice!(float*, 2LU, Contiguous) priorBox;
    size_t width = 640, height = 640;
    size_t topK;
    float minConf, nmsThreshold;
    OrtSession* session;
    OrtEnv* env;
public:
    this(int threads = 4, float minConf = 0.4, float nmsThreshold = 0.4, size_t topK = 1) {
        OrtSessionOptions* session_options;
        checkStatus(ort.CreateSessionOptions(&session_options));
        checkStatus(ort.SetInterOpNumThreads(session_options, 1));
        checkStatus(ort.SetIntraOpNumThreads(session_options, threads));
        checkStatus(ort.SetSessionExecutionMode(session_options, ExecutionMode.ORT_SEQUENTIAL));
        checkStatus(ort.SetSessionGraphOptimizationLevel(session_options, GraphOptimizationLevel.ORT_ENABLE_ALL));
        checkStatus(ort.SetSessionLogSeverityLevel(session_options, 3));
        scope(exit) {
            ort.ReleaseSessionOptions(session_options);
        }
    
	    checkStatus(ort.CreateEnv(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, "inochi-retinaface", &this.env));
	    checkStatus(ort.CreateSession(env, "models/retinaface_640x640_opt.onnx", session_options, &this.session));

        this.minConf = minConf;
        this.nmsThreshold = nmsThreshold;
        this.topK = topK;
        
        import mir.ndslice.fuse : fuse;
        priorBox = priorBoxJson.deserialize!(float[][]).sliced.fuse;
    }

    ~this() {
        ort.ReleaseSession(this.session);
        ort.ReleaseEnv(this.env);
    }

    private Slice!(float*, 2, Contiguous) decode(Slice!(float*, 2, Contiguous) loc, Slice!(float*, 2, Contiguous) priors, float[] variances) {
        import std.math.exponential : exp;

        auto first = loc[0..$, 0..2].dup;
        first[] *= variances[0];
        first[] *= priors[0..$, 2..$];
        first[] += priors[0..$, 0..2];
        
        auto second = priors[0..$, 2..$].dup;
        second[] *= loc[0..$, 2..$].map!(a => exp(a * variances[1]));

        auto boxes = concatenation!1(first, second).slice;

        boxes[0..$, 0..2] -= boxes[0..$, 2..$] / 2;
        boxes[0..$, 2..$] += boxes[0..$, 0..2];

        return boxes;
    }

    private auto submitData(Slice!(float*, 4, Contiguous) input) {
        long[] inputsDims = [1, 3, 640, 640];
        assert(input.length == inputsDims.fold!((a, b) => a * b));

        OrtMemoryInfo* memoryInfo;
	    checkStatus(ort.CreateCpuMemoryInfo(OrtAllocatorType.OrtArenaAllocator,
			OrtMemType.OrtMemTypeDefault, &memoryInfo));
        scope (exit) {
            ort.ReleaseMemoryInfo(memoryInfo);
        }

        OrtValue* inputTensor;
	    checkStatus(ort.CreateTensorWithDataAsOrtValue(memoryInfo,
			input.ptr, input.length * float.sizeof, inputsDims.ptr,
			inputsDims.length,
			ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputTensor));
        scope (exit) {
            ort.ReleaseValue(inputTensor);
        }

        {
            int isTensor;
            checkStatus(ort.IsTensor(inputTensor, &isTensor));
            assert(isTensor);
        }

        const(char)*[] inputNames = ["input0".toStringz()];
        const(char)*[] outputNames = ["output0".toStringz(), "586".toStringz()];
        OrtValue*[] outputTensor = [null, null];
        checkStatus(ort.Run(this.session, null, inputNames.ptr, &inputTensor, 1,
                outputNames.ptr, 2, outputTensor.ptr));


        foreach (i; 0..outputNames.length) {
            int isTensor;
            checkStatus(ort.IsTensor(outputTensor[i], &isTensor));
            assert(isTensor);
        }
        scope(exit) {
            foreach (i; 0..outputNames.length) {
                ort.ReleaseValue(outputTensor[i]);
            }
        }


        float* ptr;
	    
        checkStatus(ort.GetTensorMutableData(outputTensor[0], cast(void**)&ptr));
        auto firstOutput = sliced(ptr, [1, 16_800, 4])[0].dup;
	    checkStatus(ort.GetTensorMutableData(outputTensor[1], cast(void**)&ptr));
        auto secondOutput = sliced(ptr, [1, 16_800, 2])[0].dup;

        return tuple(firstOutput, secondOutput);
    }

    FaceData[] detectFaces(Image frame) {
        // Convert from Image RGB to BGR
        auto data = frame.sliced.reversed!2;

        auto resized = data.resize([this.width, this.height]).as!float.slice;        
        resized[] -= [104, 117, 123].sliced;

        auto transposed = resized.transposed!(2, 0, 1).unsqueeze!0;

        auto output = submitData(transposed.slice);
        auto loc = output[0];
        auto conf = output[1];
        
        auto boxes = decode(loc, this.priorBox, [0.1, 0.2]);
        boxes[] *= [frame.width, frame.height, frame.width, frame.height].sliced;

        auto scores = conf[0..$, 1];
        auto indices = filterIndices!((a) => a > this.minConf)(scores);

        import mir.ndslice.fuse : fuse;
        auto indBoxes = boxes.indexed(indices).fuse;
        writeln(indBoxes.shape); //[49, 4]
        auto indScores = scores.indexed(indices).fuse;

        auto keep = nms(indBoxes.fuse, indScores.fuse, this.nmsThreshold);
        auto keptDets = indBoxes[keep.sliced, 0..$];

        auto dets = keptDets[0..min(this.topK, $)].dup;
        dets[0..$, 2..4] -= dets[0..$, 0..2];

        auto upsize = dets[0..$, 2..4].dup;
        upsize[] *= [0.15, 0.2].sliced;

        dets[0..$, 0..2] -= upsize;
        dets[0..$, 2..4] += upsize * 2;

        FaceData[] ret = [];
        foreach (i; dets) {
            ret ~= FaceData(i[0], i[1], i[2], i[3]);
        }
        return ret;
    }
}

import mir.functional: naryFun;
auto filterIndices(alias pred = "a", Iterator, size_t N, SliceKind kind)(Slice!(Iterator, N, kind) r)
    if (is(typeof(naryFun!pred(r.front)))) {
    import mir.algorithm.iteration : filter;
    import mir.functional: naryFun;
    import mir.ndslice.topology: iota, ndiota;

    import std.array : array;

    static if (N == 1) {
        return r.length
            .iota
            .filter!((a) => naryFun!pred(r[a]))
            .array;
    } else {
        return r.shape
            .ndiota
            .filter!((a) => naryFun!pred(r[a]))
            .array;
    }
}
