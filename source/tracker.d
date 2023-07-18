module tracker;

import std.algorithm : any, canFind, clamp, cmp, fold, max, maxElement, min, minIndex, minElement, sort;
import std.array : array, staticArray, replicate;
import std.conv : to;
import std.math : atan2, cos, isNaN, pow, sin, sqrt, trunc, PI;
import std.range : enumerate, front, popFront;
import std.stdio;
import std.string : toStringz;
import std.typecons : tuple, Tuple;

import bindbc.onnxruntime.config;
import bindbc.onnxruntime.v12.bind;
import bindbc.onnxruntime.v12.types;

import mir.ndslice;
import mir.ndslice.topology;

import dcv.core;
import dcv.imgproc;
import dcv.imageio : imwrite;

import faceinfo;
import retinaface;
import ortdata;

import kaleidic.lubeck : inv, mtimes;
import inmath : mat3, quat, vec3;

import solvepnp : solvePnpIterative;

static immutable float[][] facePnp = [
    [ 0.4551769692672  ,  0.300895790030204, -0.764429433974752],
    [ 0.448998827123556,  0.166995837790733, -0.765143004071253],
    [ 0.437431554952677,  0.022655479179981, -0.739267175112735],
    [ 0.415033422928434, -0.088941454648772, -0.747947437846473],
    [ 0.389123587370091, -0.232380029794684, -0.704788385327458],
    [ 0.334630113904382, -0.361265387599081, -0.615587579236862],
    [ 0.263725112132858, -0.460009725616771, -0.491479221041573],
    [ 0.16241621322721 , -0.558037146073869, -0.339445180872282],
    [ 0.               , -0.621079019321682, -0.287294770748887],
    [-0.16241621322721 , -0.558037146073869, -0.339445180872282],
    [-0.263725112132858, -0.460009725616771, -0.491479221041573],
    [-0.334630113904382, -0.361265387599081, -0.615587579236862],
    [-0.389123587370091, -0.232380029794684, -0.704788385327458],
    [-0.415033422928434, -0.088941454648772, -0.747947437846473],
    [-0.437431554952677,  0.022655479179981, -0.739267175112735],
    [-0.448998827123556,  0.166995837790733, -0.765143004071253],
    [-0.4551769692672  ,  0.300895790030204, -0.764429433974752],
    [ 0.385529968662985,  0.402800553948697, -0.310031082540741],
    [ 0.322196658344302,  0.464439136821772, -0.250558059367669],
    [ 0.25409760441282 ,  0.46420381416882 , -0.208177722146526],
    [ 0.186875436782135,  0.44706071961879 , -0.145299823706503],
    [ 0.120880983543622,  0.423566314072968, -0.110757158774771],
    [-0.120880983543622,  0.423566314072968, -0.110757158774771],
    [-0.186875436782135,  0.44706071961879 , -0.145299823706503],
    [-0.25409760441282 ,  0.46420381416882 , -0.208177722146526],
    [-0.322196658344302,  0.464439136821772, -0.250558059367669],
    [-0.385529968662985,  0.402800553948697, -0.310031082540741],
    [ 0.               ,  0.293332603215811, -0.137582088779393],
    [ 0.               ,  0.194828701837823, -0.069158109325951],
    [ 0.               ,  0.103844017393155, -0.009151819844964],
    [ 0.               ,  0.               ,  0.               ],
    [ 0.080626352317973, -0.041276068128093, -0.134161035564826],
    [ 0.046439347377934, -0.057675223874769, -0.102990627164664],
    [ 0.               , -0.068753126205604, -0.090545348482397],
    [-0.046439347377934, -0.057675223874769, -0.102990627164664],
    [-0.080626352317973, -0.041276068128093, -0.134161035564826],
    [ 0.315905195966084,  0.298337502555443, -0.285107407636464],
    [ 0.275252345439353,  0.312721904921771, -0.244558251170671],
    [ 0.176394511553111,  0.311907184376107, -0.219205360345231],
    [ 0.131229723798772,  0.284447361805627, -0.234239149487417],
    [ 0.184124948330084,  0.260179585304867, -0.226590776513707],
    [ 0.279433549294448,  0.267363071770222, -0.248441437111633],
    [-0.131229723798772,  0.284447361805627, -0.234239149487417],
    [-0.176394511553111,  0.311907184376107, -0.219205360345231],
    [-0.275252345439353,  0.312721904921771, -0.244558251170671],
    [-0.315905195966084,  0.298337502555443, -0.285107407636464],
    [-0.279433549294448,  0.267363071770222, -0.248441437111633],
    [-0.184124948330084,  0.260179585304867, -0.226590776513707],
    [ 0.121155252430729, -0.208988660580347, -0.160606287940521],
    [ 0.041356305910044, -0.194484199722098, -0.096159882202821],
    [ 0.               , -0.205180167345702, -0.083299217789729],
    [-0.041356305910044, -0.194484199722098, -0.096159882202821],
    [-0.121155252430729, -0.208988660580347, -0.160606287940521],
    [-0.132325402795928, -0.290857984604968, -0.187067868218105],
    [-0.064137791831655, -0.325377847425684, -0.158924039726607],
    [ 0.               , -0.343742581679188, -0.113925986025684],
    [ 0.064137791831655, -0.325377847425684, -0.158924039726607],
    [ 0.132325402795928, -0.290857984604968, -0.187067868218105],
    [ 0.181481567104525, -0.243239316141725, -0.231284988892766],
    [ 0.083999507750469, -0.239717753728704, -0.155256465640701],
    [ 0.               , -0.256058040176369, -0.0950619498899  ],
    [-0.083999507750469, -0.239717753728704, -0.155256465640701],
    [-0.181481567104525, -0.243239316141725, -0.231284988892766],
    [-0.074036069749345, -0.250689938345682, -0.177346470406188],
    [ 0.               , -0.264945854681568, -0.112349967428413],
    [ 0.074036069749345, -0.250689938345682, -0.177346470406188],
    // Pupils and eyeball centers
    [ 0.257990002632141,  0.276080012321472, -0.219998998939991],
    [-0.257990002632141,  0.276080012321472, -0.219998998939991],
    [ 0.257990002632141,  0.276080012321472, -0.324570998549461],
    [-0.257990002632141,  0.276080012321472, -0.324570998549461]
];

enum ModelKind {
    V = -3,
    U = -2,
    T = -1,
    N0 = 0,
    N1 = 1,
    N2 = 2,
    N3 = 3,
    N4 = 4,
}

struct ResultData {
    float conf = -1;
    Slice!(float*, 2, Contiguous) lms = slice!float(0, 0);
    EyeState[] eyeState = [];
    float bonus = 0;
}

import std.traits : EnumMembers;

// for some reason static initialization doesn't work... weird
static immutable (immutable ubyte[])[ModelKind] modelData;

shared static this() {
    modelData = [
        ModelKind.V: cast(immutable ubyte[])import("lm_modelV_opt.onnx"),
        ModelKind.U: cast(immutable ubyte[])import("lm_modelU_opt.onnx"),
        ModelKind.T: cast(immutable ubyte[])import("lm_modelT_opt.onnx"),
        ModelKind.N0: cast(immutable ubyte[])import("lm_model0_opt.onnx"),
        ModelKind.N1: cast(immutable ubyte[])import("lm_model1_opt.onnx"),
        ModelKind.N2: cast(immutable ubyte[])import("lm_model2_opt.onnx"),
        ModelKind.N3: cast(immutable ubyte[])import("lm_model3_opt.onnx"),
        ModelKind.N4: cast(immutable ubyte[])import("lm_model4_opt.onnx")
    ];
}
// V 1 198 14 14
// U 1 198 14 14
// T 1 90 7 7
// N0 1 198 28 28
// N1 1 198 28 28
// N2 1 198 28 28
// N3 1 198 28 28
// N4 1 198 28 28

struct TrackerSettings {
    ModelKind kind = ModelKind.N3;

    float threshold = 0.0;
    float detectionThreshold = 0.6;

    int maxFeatureUpdates = 0;
    int maxFaces = 1;
    int maxThreads = 4;

    int featureLevel = 2;
    int discardAfter = 5;
    int scanEvery = 3;

    float bboxGrowth = 0.0;
    
    bool noGaze = false;
    bool useRetinaface = false;
    bool tryHard = false;
    bool staticModel = false;
    bool debugGaze = false;
}

struct BoundingBox {
    Point point;
    float width;
    float height;
}

struct EyeState {
    bool open;
    Point p;
    float conf;
}

class Tracker {
private:
    alias CropData = Tuple!(
        Slice!(float*, 4, Contiguous), float, "x1", float, "y1", float, "scaleX", float, "scaleY", float, "bonus"
    );
    alias CropDetails = Tuple!(
        float, "x1", float, "y1", float, "scaleX", float, "scaleY", float, "bonus"
    );

    Slice!(float*, 1, SliceKind.contiguous) mean, std;
    Slice!(float*, 3, SliceKind.contiguous) mean32, std32, mean224, std224;

    int res, out_res_i;
    float logitFactor;
    Slice!(float*, 3, SliceKind.contiguous) meanRes, stdRes;

    OrtSession* session, detection, gaze;
    OrtEnv* env;
    RetinaFace retinaFace, retinaFaceScan;

    size_t width, height;
    int frameCount = 0;
    int waitCount = 0;
    int numDetected = 0;
    int discard = 0;

    TrackerSettings settings;

    FaceInfo[] faceInfos = [];
    FaceData[] faces = [];
public:
    this(size_t width, size_t height, TrackerSettings settings = TrackerSettings()) {
        OrtSessionOptions* sessionOptions;
        checkStatus(ort.CreateSessionOptions(&sessionOptions));
        scope(exit) {
            ort.ReleaseSessionOptions(sessionOptions);
        }
        checkStatus(ort.SetInterOpNumThreads(sessionOptions, 1));
        checkStatus(ort.SetIntraOpNumThreads(sessionOptions, max(settings.maxThreads, 4)));
        checkStatus(ort.SetSessionExecutionMode(sessionOptions, ExecutionMode.ORT_SEQUENTIAL));
        checkStatus(ort.SetSessionGraphOptimizationLevel(sessionOptions, GraphOptimizationLevel.ORT_ENABLE_ALL));
        checkStatus(ort.SetSessionLogSeverityLevel(sessionOptions, 3));

	    checkStatus(ort.CreateEnv(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, "inochi-osf", &this.env));
    
        {
            OrtSessionOptions* tempOptions;
            checkStatus(ort.CloneSessionOptions(sessionOptions, &tempOptions));
            scope(exit) {
                ort.ReleaseSessionOptions(tempOptions);
            }
            checkStatus(ort.AddFreeDimensionOverride(tempOptions, "batch_size".toStringz, 1));
            
            auto model = modelData[this.settings.kind];
            checkStatus(ort.CreateSessionFromArray(this.env, model.ptr, model.length, tempOptions, &this.session));

            auto detectionModel = import("mnv3_detection_opt.onnx");
            checkStatus(ort.CreateSessionFromArray(this.env, detectionModel.ptr, detectionModel.length, sessionOptions, &this.detection));
        }

        {
            OrtSessionOptions* tempOptions;
            checkStatus(ort.CloneSessionOptions(sessionOptions, &tempOptions));
            scope(exit) {
                ort.ReleaseSessionOptions(tempOptions);
            }
            checkStatus(ort.AddFreeDimensionOverride(tempOptions, "batch_size".toStringz, 2));
            
            auto gazeModel = import("mnv3_gaze32_split_opt.onnx");
	        checkStatus(ort.CreateSessionFromArray(this.env, gazeModel.ptr, gazeModel.length, sessionOptions, &this.gaze));
        }

        auto mean = [0.485, 0.456, 0.406].sliced!float;
        auto std = [0.229, 0.224, 0.225].sliced!float;
        mean[] /= std;
        std[] *= 255.0;

        mean[] = -mean[];
        std[] = 1.0 / std[];

        this.mean = mean;
        this.std = std;

        this.mean32 = mean.repeat(32, 32).fuse;
        this.std32 = std.repeat(32, 32).fuse;
        this.mean224 = mean.repeat(224, 224).fuse;
        this.std224 = std.repeat(224, 224).fuse;

        this.width = width;
        this.height = height;

        this.settings = settings;

        this.res = 224;
        this.out_res_i = 28;
        this.logitFactor = 16;
        this.meanRes = this.mean224;
        this.stdRes = this.std224;
        if (this.settings.kind < 0) {
            this.res = 56;
            this.out_res_i = 7;
            this.logitFactor = 8;
            this.meanRes = mean.repeat(56, 56).fuse;
            this.stdRes = std.repeat(56, 56).fuse;
        }
        if (this.settings.kind < -1) {
            this.res = 112;
            this.out_res_i = 14;
            this.logitFactor = 16;
            this.meanRes = mean.repeat(112, 112).fuse;
            this.stdRes = std.repeat(112, 112).fuse;
        }

        foreach (i; 0..settings.maxFaces) {
            this.faceInfos ~= new FaceInfo(this, this.settings.kind);
        }
    }

    ~this() {
        ort.ReleaseSession(this.detection);
        ort.ReleaseSession(this.session);
        ort.ReleaseEnv(this.env);
    }

    private auto submitDetection(Slice!(float*, 4, Contiguous) input) {
        long[] inputsDims = [1, 3, 224, 224];
        assert(input.elementCount == inputsDims.fold!((a, b) => a * b));

        OrtMemoryInfo* memoryInfo;
	    checkStatus(ort.CreateCpuMemoryInfo(OrtAllocatorType.OrtArenaAllocator,
			OrtMemType.OrtMemTypeDefault, &memoryInfo));
        scope (exit) {
            ort.ReleaseMemoryInfo(memoryInfo);
        }

        OrtValue* inputTensor;
	    checkStatus(ort.CreateTensorWithDataAsOrtValue(memoryInfo,
			input.ptr, input.elementCount * float.sizeof, inputsDims.ptr,
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

        const(char)*[] inputNames = ["input".toStringz()];
        const(char)*[] outputNames = ["output".toStringz(), "maxpool".toStringz()];
        OrtValue*[] outputTensor = [null, null];
        checkStatus(ort.Run(this.detection, null, inputNames.ptr, &inputTensor, 1,
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
        auto firstOutput = sliced(ptr, [1, 2, 56, 56])[0].dup;
	    checkStatus(ort.GetTensorMutableData(outputTensor[1], cast(void**)&ptr));
        auto secondOutput = sliced(ptr, [1, 2, 56, 56])[0].dup;

        return tuple(firstOutput, secondOutput);
    }

    FaceData[] detectFaces(Image frame) {
        assert(frame.format == ImageFormat.IF_RGB);
        auto im = frame.sliced().resize([224, 224]) * this.std224[] + this.mean224[];
        auto unsqueezed = im.unsqueeze!0;
        auto transposed = unsqueezed.transposed!(0, 3, 1, 2);

        auto results = this.submitDetection(transposed.fuse);
        auto outputs = results[0];
        auto maxpool = results[1];
        
        auto eq = zip(outputs[0], maxpool[0]).map!((a, b) => a != b);
        outputs[0].indexed(filterIndices(eq).array)[] = 0;

        import mir.ndslice.sorting : makeIndex;
        auto detections = makeIndex(outputs[0].flattened).reversed!0;

        FaceData[] ret = [];
        foreach (det; detections[0..max($, this.settings.maxFaces)]) {
            auto y = det / 56;
            auto x = det % 56;

            auto c = outputs[0, y, x];
            auto r = outputs[1, y, x] * 112;

            x *= 4;
            y *= 4;

            if (c < this.settings.detectionThreshold) {
                break;
            }

            ret ~= FaceData(
                (x - r) * frame.width / 224,
                (y - r) * frame.height / 224,
                (2 * r) * frame.width / 224,
                (2 * r) * frame.height / 224
            );
        }
        return ret;
    }

    Slice!(float*, 4, Contiguous) preprocess(Image frame, int x1, int y1, int x2, int y2) {
        assert(frame.format == ImageFormat.IF_RGB);
        auto cropped = frame.sliced;

        auto resized = cropped[y1..y2, x1..x2, 0..$].resize([this.res, this.res]).as!float.slice;
        auto unsqueezed = resized.unsqueeze!0;
        auto transposed = unsqueezed.transposed(0, 3, 1, 2);

        return transposed.slice;
    }

    private auto submitSession(Slice!(float*, 4, Contiguous) input) {
        long[] inputsDims = [1, 3, this.res, this.res];
        assert(input.elementCount == inputsDims.fold!((a, b) => a * b));

        OrtMemoryInfo* memoryInfo;
	    checkStatus(ort.CreateCpuMemoryInfo(OrtAllocatorType.OrtArenaAllocator,
			OrtMemType.OrtMemTypeDefault, &memoryInfo));
        scope (exit) {
            ort.ReleaseMemoryInfo(memoryInfo);
        }

        OrtValue* inputTensor;
	    checkStatus(ort.CreateTensorWithDataAsOrtValue(memoryInfo,
			input.ptr, input.elementCount * float.sizeof, inputsDims.ptr,
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

        const(char)*[] inputNames = ["input".toStringz()];
        const(char)*[] outputNames = ["output".toStringz()];
        OrtValue* outputTensor = null;
        checkStatus(ort.Run(this.session, null, inputNames.ptr, &inputTensor, 1,
                outputNames.ptr, 1, &outputTensor));

        {
            int isTensor;
            checkStatus(ort.IsTensor(outputTensor, &isTensor));
            assert(isTensor);
        }
        scope(exit) {
            ort.ReleaseValue(outputTensor);
        }

        float* ptr;
        checkStatus(ort.GetTensorMutableData(outputTensor, cast(void**)&ptr));

        auto output = sliced(ptr, [1, 198, 28, 28])[0].dup;

        return output;
    }

    auto landmarks(Slice!(float*, 3, Contiguous) tensor, CropDetails cropInfo) {
        import fghj;
        tensor = import("test3.json").deserialize!(float[][][]).sliced.fuse;
        cropInfo = typeof(cropInfo)(289, 220, 2.924107142857143, 1.7142857142857142, 0.0);

        auto res = this.res - 1;
        auto c0 = this.settings.kind == ModelKind.T ? 30 : 66;
        auto c1 = this.settings.kind == ModelKind.T ? 60 : 132;
        auto c2 = this.settings.kind == ModelKind.T ? 90 : 198;

        auto reshapeDims = [c0, this.out_res_i * this.out_res_i].staticArray!ptrdiff_t;

        int err;
        auto reshaped = tensor[0..c0].reshape(reshapeDims, err);
        assert(err == 0);

        auto maxInds = reshaped.byDim!0.map!((a) => a.maxIndex[0]).fuse;

        // no `take_along_axis`, so there's going to be a lot of zips and maps here
        auto tConf = reshaped.byDim!0.zip(maxInds).map!((a, b) => a[b]);

        auto offX = tensor[c0..c1].reshape(reshapeDims, err).byDim!0.zip(maxInds).map!((a, b) => a[b]);
        assert(err == 0);
        auto offY = tensor[c1..c2].reshape(reshapeDims, err).byDim!0.zip(maxInds).map!((a, b) => a[b]);
        assert(err == 0);

        auto logitX = logitSlice(offX, this.logitFactor) * res;
        auto logitY = logitSlice(offY, this.logitFactor) * res;
        
        float adjustedOutRes = this.out_res_i - 1;
        auto tX = (res * (maxInds / this.out_res_i) / adjustedOutRes + logitX) * cropInfo.scaleY + cropInfo.y1;
        auto tY = (res * (maxInds % this.out_res_i) / adjustedOutRes + logitY) * cropInfo.scaleX + cropInfo.x1;

        import mir.math.stat : mean;
        float avgConf = mean(tConf);

        auto lms = [tX.slice, tY.slice, tConf.slice].fuse!1;
        foreach (i; lms.byDim!0) {
            if (any!(isNaN)(i)) {
                i[] = [0.0, 0.0, 0.0];
            }
        }
        
        if (this.settings.kind == ModelKind.T) {
            immutable static size_t[] lmsIndices = [
                0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 7, 7, 8,
                8, 9, 10, 10, 11, 11, 12, 21, 21, 21, 22, 23, 23, 23, 23, 23, 13, 14, 14, 15, 16,
                16, 17, 18, 18, 19, 20, 20, 24, 25, 25, 25, 26, 26, 27, 27, 27,
                24, 24, 28, 28, 28, 26, 29, 29, 29
            ];

            // TODO: all of this
            throw new Exception("not yet implemented");
        }

        return tuple!("avgConf", "lms")(avgConf, lms);
    }

    FaceData[] predict(Image frame) {
        assert(frame.format == ImageFormat.IF_RGB);
        this.frameCount++;
        this.waitCount++;

        FaceData[] newFaces = this.faces.dup;
        
        auto bonusCutoff = this.faces.length;
        // if (this.numDetected == 0) {
        //     if (this.settings.useRetinaface || this.settings.tryHard) {
        //         newFaces ~= this.retinaFace.detectFaces(frame);

        //         if (this.settings.tryHard) {
                    newFaces ~= this.detectFaces(frame);
        //         }
        //     }

        //     if (this.settings.tryHard) {
                // newFaces ~= FaceData(0, 0, this.width, this.height);
        //     }
        // } else if (this.numDetected < this.settings.maxFaces) {
        //     if (this.settings.useRetinaface) {
        //         // new_faces.extend(self.retinaface_scan.get_results())
        //     }

        //     if (this.waitCount >= this.settings.scanEvery) {
        //         if (this.settings.useRetinaface) {
        //             // self.retinaface_scan.background_detect(frame)
        //         } else {
        //             newFaces ~= this.detectFaces(frame);
                    
        //             // TODO: is this supposed to be in the else?
        //             this.waitCount = 0;
        //         }
        //     }
        // } else {
        //     this.waitCount = 0;
        // }

        if (newFaces.length == 0) {
            return [];
        }

        CropData[] crops = [];
        foreach (i, face; newFaces) {
            auto cropX1 = face.x - to!int(face.width * 0.1);
            auto cropY1 = face.y - to!int(face.height * 0.125);
            auto cropX2 = face.x + face.width + to!int(face.width * 0.1);
            auto cropY2 = face.y + face.height + to!int(face.height * 0.125);

            cropX1 = clamp(cropX1, 0, this.width).trunc;
            cropY1 = clamp(cropY1, 0, this.height).trunc + 1;
            cropX2 = clamp(cropX2, 0, this.width).trunc;
            cropY2 = clamp(cropY2, 0, this.height).trunc + 1;

            if (cropX2 - cropX1 < 4 || cropY2 - cropY1 < 4) {
                continue;
            }

            auto cropped = this.preprocess(frame, to!int(cropX1), to!int(cropY1), to!int(cropX2), to!int(cropY2));

            auto scaleX = (cropX2 - cropX1) / this.res;
            auto scaleY = (cropY2 - cropY1) / this.res;

            crops ~= CropData(cropped, cropX1, cropY1, scaleX, scaleY, i >= bonusCutoff ? 0.0f : 0.1f);
        }

        struct OutputData {
            float conf;
            Slice!(float*, 2, Contiguous) lms;
            EyeState[] eyeState;
            BoundingBox bb;
        }

        OutputData[CropDetails] outputs;
        if (crops.length == 1) {
            auto results = this.submitSession(crops.front[0]);
            auto details = crops.front.slice!(1, CropData.Types.length);
            auto data = this.landmarks(results, details);

            if (data.avgConf >= this.settings.threshold) {
                auto eyeState = this.getEyeState(frame, data.lms);

                auto lms = data.lms;
                auto mins = lms.byDim!1.map!((a) => a.minElement);
                auto maxes = lms.byDim!1.map!((a) => a.maxElement);
                auto bb = BoundingBox(Point(mins[0], mins[1]), maxes[0] - mins[0], maxes[1] - mins[1]);

                outputs[details] = OutputData(data.avgConf, data.lms, eyeState, bb);
            }
        } else {
            throw new Exception("not implemented");
        }

        BoundingBox[] actualFaces;
        CropDetails[] goodCrops;

        foreach (crop; crops) {
            auto details = crop.slice!(1, CropData.Types.length);
            auto data = details in outputs;
            if (data is null) {
                continue;
            }

            actualFaces ~= data.bb;
            goodCrops ~= details;
        }

        auto groups = groupRects(actualFaces);

        ResultData[int] bestResults;

        foreach (crop; goodCrops) {
            auto data = outputs[crop];
            if (data.conf < this.settings.threshold) {
                continue;
            }

            int groupId = groups[data.bb];

            bestResults.require(groupId);
            
            if (bestResults[groupId].conf < data.conf + crop.bonus) {
                bestResults[groupId] = ResultData(data.conf + crop.bonus, data.lms, data.eyeState, crop.bonus);
            }
        }

        auto sortedResults = bestResults.byValue.array;
        sortedResults.sort!("a.conf > b.conf");
        assignFaceInfo(sortedResults[0..min($, this.settings.maxFaces)]);

        foreach(faceInfo; this.faceInfos) {
            if (faceInfo.alive && faceInfo.conf > this.settings.threshold) {
                auto res = this.estimateDepth(faceInfo);
                faceInfo.pnpError = res.pnpError;
                faceInfo.lms = res.lms;
                faceInfo.adjust3d(res.points3d, this.settings.kind, this.settings.staticModel);
            }
        }


        return [];
    }

    struct EstimateResult {
        float pnpError;
        Slice!(float*, 2, Contiguous) points3d;
        Slice!(float*, 2, Contiguous) lms;
    }

    Slice!(float*, 2, Contiguous) camera() {
        return [[this.width, 0f, this.width / 2f], [0f, this.width, this.height / 2f], [0f, 0f, 1f]].fuse;
    }

    EstimateResult estimateDepth(FaceInfo info) {
        auto lms = concatenation(
            info.lms,
            [[info.eyeState[0].p.x, info.eyeState[0].p.y, info.eyeState[0].conf],
             [info.eyeState[1].p.x, info.eyeState[1].p.y, info.eyeState[1].conf]].sliced.fuse, 
        ).slice;

        auto imagePts = lms[info.contourPoints, 0..2].slice;

        auto invCamera = camera.inv;
    
        try {
            if (info.rotation.isFinite) {
                auto a = solvePnpIterative(info.contour, imagePts, camera, slice!float(0),
                    info.rotation.toAxisAngle.vector.sliced, info.translation.vector.sliced);

                vec3 rot = vec3(a.rvec[0], a.rvec[1], a.rvec[2]);
                info.rotation = quat.axisRotation(rot.length, rot.normalized);
                info.translation = vec3(a.tvec[0], a.tvec[1], a.tvec[2]);
            } else {
                auto rvec = [0f, 0f, 0f].sliced;
                auto tvec = [0f, 0f, 0f].sliced;
                auto a = solvePnpIterative(info.contour, imagePts, camera, slice!float(0), rvec, tvec);

                vec3 rot = vec3(a.rvec[0], a.rvec[1], a.rvec[2]);
                info.rotation = quat.axisRotation(rot.length, rot.normalized);
                info.translation = vec3(a.tvec[0], a.tvec[1], a.tvec[2]);
            }
        } catch (Exception e) {
            // Do something here
        }

        auto points3d = slice!float([70, 3], 0);

        auto rmat = info.rotation.toMatrix!(3, 3);
        auto invRmat = rmat.inverse;

        auto rmatSlice = sliced(rmat.ptr, 3, 3);
        auto invRmatSlice = sliced(invRmat.ptr, 3, 3);
        auto tRef = mtimes(info.face3d, rmatSlice.transposed);
        auto translationSlice = info.translation.vector.sliced;
        
        tRef[] += translationSlice;
        tRef = mtimes(tRef, camera.transposed);


        auto tDepth = tRef[0..$, 2];
        tDepth.indexed(tDepth.filterIndices!"a == 0".array)[] = 0.000001;

        // This is a matrix [70, 3] / [70, 1] which just doesn't work at all
        // in ndslice so I have to repeat it in place and that causes an allocation
        // but oh well.
        auto tDepthM = tDepth.map!"[a, a, a]".fuse;
        tRef[] /= tDepthM[];

        auto concatted = concatenation!1(lms[0..66, 0].unsqueeze!1, lms[0..66, 1].unsqueeze!1, repeat(1, 66).unsqueeze!1);

        points3d[0..66, 0..$] = concatted[];
        points3d[0..66, 0..$] *= tDepthM[0..66];

        auto dotted = mtimes(points3d[0..66, 0..$], invCamera.transposed);
        dotted[0..$, 0..$] -= translationSlice;
        
        points3d[0..66, 0..$] = mtimes(dotted, invRmatSlice.transposed);

        auto pnpError = 0.0f;

        import mir.math.sum : sum;
        {
            auto errorCalc = lms[0..17, 0..2].dup;
            errorCalc[] -= tRef[0..17, 0..2];
            auto s = errorCalc.flattened.map!"pow(a, 2)";
            pnpError += s.sum;
        }
        
        {
            auto errorCalc = lms[30, 0..2].dup;
            errorCalc[] -= tRef[30, 0..2];
            auto s = errorCalc.flattened.map!"pow(a, 2)";
            pnpError += s.sum;
        }

        if (pnpError.isNaN) {
            pnpError = 9_999_999.0f;
        }

        foreach (i, _; info.face3d[66..70].enumerate) {
            // Right eyeball
            if (i == 2) {
                auto eyeCenter = (points3d[36] + points3d[39]) / 2.0;
                auto subbed = points3d[36] - points3d[39];
                auto dCorner = norm(subbed[0], subbed[1]);
                auto depth = 0.385 * dCorner;
                auto point3d = [eyeCenter[0], eyeCenter[1], eyeCenter[2] - depth];
                points3d[68][] = point3d;
                continue;
            }

            // Left eyeball
            if (i == 3) {
                auto eyeCenter = (points3d[42] + points3d[45]) / 2.0;
                auto subbed = points3d[42] - points3d[45];
                auto dCorner = norm(subbed[0], subbed[1]);
                auto depth = 0.385 * dCorner;
                auto point3d = [eyeCenter[0], eyeCenter[1], eyeCenter[2] - depth];
                points3d[69][] = point3d;
                continue;
            }

            auto pt = slice!float(3);
            if (i == 0) {
                auto subbed1 = lms[66] - lms[36];
                auto d1 = norm(subbed1[0], subbed1[1]);
                auto subbed2 = lms[66] - lms[39];
                auto d2 = norm(subbed2[0], subbed2[1]);

                auto d = d1 + d2;
                pt[] = (points3d[36] * d1 + points3d[39] * d2) / d;
            }
            if (i == 1) {
                auto subbed1 = lms[67] - lms[42];
                auto d1 = norm(subbed1[0], subbed1[1]);
                auto subbed2 = lms[67] - lms[45];
                auto d2 = norm(subbed2[0], subbed2[1]);

                auto d = d1 + d2;
                pt[] = (points3d[42] * d1 + points3d[45] * d2) / d;
            }

            if (i < 2) {
                auto reference = mtimes(rmatSlice, pt);
                reference[] += translationSlice;
                reference[] = mtimes(camera, reference);

                auto depth = reference[2];
                auto point3d = [lms[66 + i][0] * depth, lms[66 + i][1] * depth, depth].sliced;
                point3d[] = mtimes(invCamera, point3d);
                point3d[] -= translationSlice;
                point3d[] = mtimes(invRmatSlice, point3d);
                points3d[66 + i][] = point3d;
            }
        }

        foreach (i; points3d.byDim!0) {
            if (any!(isNaN)(i)) {
                i[] = [0.0, 0.0, 0.0];
            }
        }
        
        pnpError = sqrt(pnpError / (2.0 * imagePts.shape[0]));
        
        if (pnpError > 300.0) {
            info.failureCount += 1;

            if (info.failureCount > 5) {
                writeln("warning: very high amounts of error");
                info.resetFace3d();
                info.rotation = quat(float.nan, float.nan, float.nan, float.nan);
                info.translation = vec3(0.0, 0.0, 0.0);
                info.updateCounts = slice!float([66, 2], 0);
                info.updateContour();
            }
        } else {
            info.failureCount = 0;
        }

        return EstimateResult(
            pnpError,
            points3d,
            lms
        );
    }

    void assignFaceInfo(ResultData[] results) {
        import mir.math.stat : mean;
        if (this.settings.maxFaces == 1 && results.length == 1) {
            auto rawCoords = results[0].lms[0..$, 0..2].alongDim!0.map!mean.fuse;
            this.faceInfos[0].update(this.frameCount, results[0], Point(rawCoords[0], rawCoords[1]));
            return;
        }

        Point[] resultCoords = [];
        Tuple!(float, Slice!(float*, 2, Contiguous), EyeState[])[] adjustedResults = [];
        
        foreach (result; results) {
            auto temp = result.lms[0..$, 0..2].alongDim!0.map!mean.fuse;
            resultCoords ~= Point(temp[0], temp[1]);
            adjustedResults ~= tuple(result.conf - result.bonus, result.lms, result.eyeState);
        }

        auto maxDist = 2 * norm(this.width, this.height);
        auto candidates = new Tuple!(float, size_t, size_t)[][this.settings.maxFaces];
        foreach (i, faceInfo; this.faceInfos) {
            foreach (j, coord; resultCoords) {
                if (!faceInfo.coord.isFinite) {
                    candidates[i] ~= tuple(maxDist, i, j);
                } else {
                    auto c = faceInfo.coord - coord;
                    candidates[i] ~= tuple(norm(c.x, c.y), i, j);
                }
            }
        }

        auto found = 0;
        auto target = results.length;

        bool[size_t] usedFaces;
        bool[size_t] usedResults;

        while (found < target) {
            auto minList = candidates[candidates.minIndex!"cmp(a, b)"];
            auto candidate = minList.front;
            minList.popFront;

            auto faceIdx = candidate[1];
            auto resultIdx = candidate[2];

            if (faceIdx !in usedFaces && resultIdx !in usedResults) {
                auto rawCoords = results[resultIdx].lms[0..$, 0..2].alongDim!0.map!mean.fuse;
                this.faceInfos[faceIdx].update(this.frameCount, results[resultIdx], Point(rawCoords[0], rawCoords[1]));
                minList.length = 0;

                usedFaces[faceIdx] = true;
                usedResults[resultIdx] = true;
            }

            if (minList.length == 0) {
                minList ~= tuple(2 * maxDist, faceIdx, resultIdx);
            }
        }

        foreach (faceInfo; this.faceInfos) {
            if (faceInfo.frameCount != this.frameCount) {
                faceInfo.update(this.frameCount);
            }
        }
    }

    struct EyeCornerData {
        int upperLeftXi, upperLeftYi;
        int lowerRightXi, lowerRightYi;
        float centerX, centerY;
        float radiusX, radiusY;
        float referenceX, referenceY;
        float angle;
    }

    EyeCornerData cornersToEye(float cX1, float cY1, float cX2, float cY2, size_t width, size_t height, bool flip) {
        auto ret = compensate(cX1, cY1, cX2, cY2);
        auto c2 = ret[0];
        auto angle = ret[1];
        
        auto centerX = (cX1 + c2.x) / 2.0;
        auto centerY = (cY1 + c2.y) / 2.0;

        auto radius = norm(cX1 - c2.x, cY1 - c2.y) / 2.0;
        auto radiusX = radius * 1.4;
        auto radiusY = radius * 1.2;

        int upperLeftXi = clamp(centerX - radiusX, 0, this.width).to!int;
        int upperLeftYi = clamp(centerY - radiusY, 0, this.height).to!int + 1;
        int lowerRightXi = clamp(centerX + radiusX, 0, this.width).to!int;
        int lowerRightYi = clamp(centerY + radiusY, 0, this.height).to!int + 1;

        EyeCornerData data = {
            upperLeftXi, upperLeftYi,
            lowerRightXi, lowerRightYi,
            centerX, centerY,
            radiusX, radiusY,
            cX1, cY1,
            angle,
        };

        return data;
    }
    
    struct EyeData {
        Slice!(float*, 4, Universal) image;
        float scaleX, scaleY; 
        int upperLeftXi, upperLeftYi;
        float referenceX, referenceY;
        float angle;
    }

    EyeData prepareEye(SliceKind frameKind)
        (Slice!(ubyte*, 3, frameKind) frame, Image fullFrame, Slice!(int*, 2, Contiguous) lms, bool flip){
        auto eye = this.cornersToEye(lms[0][0], lms[0][1], lms[1][0], lms[1][1], frame.shape[0], frame.shape[1], flip);

        auto rotated = rotateImage(frame, eye.angle, Point(eye.referenceX, eye.referenceY));
        auto sliced = rotated[eye.upperLeftYi..eye.lowerRightYi, eye.upperLeftXi..eye.lowerRightXi, 0..$];

        if (sliced.elementCount == 0) {
            return EyeData.init;
        }

        auto resized = flip ? sliced.reversed!1.resize([32, 32]) : sliced.resize([32, 32]);

        if (this.settings.debugGaze) {
            if (!flip) {
                fullFrame.sliced[0..32, 0..32][] = resized;
            } else {
                fullFrame.sliced[0..32, 32..64][] = resized;
            }
        }

        auto finalImg = resized.reversed!2.as!float.dup;
        finalImg[] *= this.std32;
        finalImg[] += this.mean32;

        auto unsqueezed = finalImg.unsqueeze!0;
        auto transposed = unsqueezed.transposed(0, 3, 2, 1);

        float scaleX = (eye.lowerRightXi - eye.upperLeftXi) / 32.0;
        float scaleY = (eye.lowerRightYi - eye.upperLeftYi) / 32.0;

        EyeData data = {
            transposed, scaleX, scaleY, eye.upperLeftXi, eye.upperLeftYi, eye.referenceX, eye.referenceY, eye.angle,
        };

        return data;
    }

    private auto submitGaze(Slice!(float*, 4, Contiguous) input) {
        long[] inputsDims = [2, 3, 32, 32];
        assert(input.elementCount == inputsDims.fold!((a, b) => a * b));

        OrtMemoryInfo* memoryInfo;
	    checkStatus(ort.CreateCpuMemoryInfo(OrtAllocatorType.OrtArenaAllocator,
			OrtMemType.OrtMemTypeDefault, &memoryInfo));
        scope (exit) {
            ort.ReleaseMemoryInfo(memoryInfo);
        }

        OrtValue* inputTensor;
	    checkStatus(ort.CreateTensorWithDataAsOrtValue(memoryInfo,
			input.ptr, input.elementCount * float.sizeof, inputsDims.ptr,
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

        const(char)*[] inputNames = ["input".toStringz()];
        const(char)*[] outputNames = ["output".toStringz(), "blink".toStringz()];
        OrtValue*[] outputTensor = [null, null];
        checkStatus(ort.Run(this.gaze, null, inputNames.ptr, &inputTensor, 1,
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
        auto firstOutput = sliced(ptr, [2, 3, 8, 8]).dup;
	    checkStatus(ort.GetTensorMutableData(outputTensor[1], cast(void**)&ptr));
        auto secondOutput = sliced(ptr, [2, 2, 1, 1]).dup;

        return tuple(firstOutput, secondOutput);
    }

    EyeState[] getEyeState(Image frame, Slice!(float*, 2, Contiguous) lms) {
        if (this.settings.noGaze) {
            return [EyeState(true, Point(0.0, 0.0), 0.0), EyeState(true, Point(0.0, 0.0), 0.0)];
        }

        assert(frame.format == ImageFormat.IF_RGB);

        auto extracted = this.extractFace(frame, lms);
        auto faceFrame = extracted.frame;
        auto faceLms = extracted.lms;
        auto offset = extracted.offset;

        auto right = this.prepareEye(faceFrame, frame, faceLms.indexed([36, 39]).fuse, false);
        auto left = this.prepareEye(faceFrame, frame, faceLms.indexed([42, 45]).fuse, true);

        if (left.angle.isNaN || right.angle.isNaN) {
            return [EyeState(true, Point(0.0, 0.0), 0.0), EyeState(true, Point(0.0, 0.0), 0.0)];
        }

        auto eX = [right.upperLeftXi, left.upperLeftXi];
        auto eY = [right.upperLeftYi, left.upperLeftYi];
        auto scale = [[right.scaleX, right.scaleY], [left.scaleX, left.scaleY]];
        auto reference = [Point(right.referenceX, right.referenceY), Point(left.referenceX, left.referenceY)];
        auto angles = [right.angle, left.angle];

        auto bothEyes = concatenation(right.image, left.image).slice;
        import fghj;
        bothEyes = import("test4.json").deserialize!(float[][][][]).sliced.fuse;

        // The second arg is just... unused?
        auto result = this.submitGaze(bothEyes)[0];

        EyeState[] eyeState = [];

        foreach (i; 0..2) {
            auto m = result[i, 0].maxIndex;
            auto conf = result[i, 0][m];

            auto x = m[0];
            auto y = m[1];

            auto offX = 32.0f * logit(result[i, 1][m], 8.0);
            auto offY = 32.0f * logit(result[i, 2][m], 8.0);

            auto eyeX = 32.0f * x.to!float / 8.0f + offX;
            auto eyeY = 32.0f * y.to!float / 8.0f + offY;

            if (this.settings.debugGaze) {
                ubyte[] fill = [0, 0, 255];
                if (i == 0) {
                    frame.sliced[eyeY.to!int, eyeX.to!int, 0..$] = fill;
                    frame.sliced[eyeY.to!int + 1, eyeX.to!int, 0..$] = fill;
                    frame.sliced[eyeY.to!int + 1, eyeX.to!int + 1, 0..$] = fill;
                    frame.sliced[eyeY.to!int, eyeX.to!int + 1, 0..$] = fill;
                } else {
                    frame.sliced[eyeY.to!int, 32 + eyeX.to!int, 0..$] = fill;
                    frame.sliced[eyeY.to!int + 1, 32 + eyeX.to!int, 0..$] = fill;
                    frame.sliced[eyeY.to!int + 1, 32 + eyeX.to!int + 1, 0..$] = fill;
                    frame.sliced[eyeY.to!int, 32 + eyeX.to!int + 1, 0..$] = fill;
                }
            }

            if (i == 0) {
                eyeX = eX[i] + scale[i][0] * eyeX;
            } else {
                eyeX = eX[i] + scale[i][0] * (32.0f - eyeX);
            }
                
            eyeY = eY[i] + scale[i][1] * eyeY;
            auto eyeRotated = rotate(reference[i], Point(eyeX, eyeY), -angles[i]);

            eyeState ~= EyeState(true, Point(eyeRotated.y + offset[1], eyeRotated.x + offset[0]), conf);
        }
        
        return eyeState;
    }

    auto extractFace(Image frame, Slice!(float*, 2, Contiguous) lms) {
        auto filtered = lms[0..$, 0..2].reversed!1;

        auto xIndices = filtered[0..$, 0..1].squeeze!1.minmaxIndex;
        auto yIndices = filtered[0..$, 1..2].squeeze!1.minmaxIndex;

        auto x1 = filtered[xIndices[0]][0];
        auto x2 = filtered[xIndices[1]][0];
        auto y1 = filtered[yIndices[0]][1];
        auto y2 = filtered[yIndices[1]][1];

        auto radiusX = 1.2 * (x2 - x1) / 2.0;
        auto radiusY = 1.2 * (y2 - y1) / 2.0;
        
        auto centerX = (x1 + x2) / 2.0;
        auto centerY = (y1 + y2) / 2.0;
        
        int x1i = clamp(centerX - radiusX, 0, this.width).to!int;
        int y1i = clamp(centerY - radiusY, 0, this.height).to!int + 1;
        int x2i = clamp(centerX + radiusX + 1, 0, this.width).to!int;
        int y2i = clamp(centerY + radiusY + 1, 0, this.height).to!int + 1;

        auto offset = [x1i, y1i].sliced;
        auto data = filtered.as!int.slice;
        data[] -= offset;

        // TODO: fix this BGR RGB mess
        auto slicedFrame = frame.sliced[y1i..y2i, x1i..x2i].reversed!2;

        return tuple!("frame", "lms", "offset")(slicedFrame, data, offset);
    }
}

float logit(float a, float factor = 16.0) {
    import std.math.exponential : log;
    a = clamp(a, 0.0000001, 0.999999);
    a = a / (1 - a);
    return log(a) / factor;
}

auto logitSlice(Iterator, SliceKind kind)(Slice!(Iterator, 1, kind) slice, float factor = 16.0)
    if (is(slice.DeepElement == float)) {
    // does this vectorize, or can I just do `slice.map!((a) => logit(a, factor))`?
    import std.math.exponential : log;
    auto clipped = slice.map!((a) => clamp(a, 0.0000001f, 0.9999999f));
    auto corresponding = clipped / (1.0f - clipped);
    auto logged = corresponding.map!((a) => log(a).to!float);
    return logged / factor;
}

auto compensate(float x1, float y1, float x2, float y2) {
    auto a = angle(x1, y1, x2, y2);
    return tuple(rotate(Point(x1, y1), Point(x2, y2), a), a);
}

// Not sure if this is needed but better
// safe than sorry
float pymod(float x, float y) {
	return ((x % y) + y) % y;
}

float angle(float x1, float y1, float x2, float y2) {
    float first = y2 - y1;
    float second = x2 - x1;

    auto angle = atan2(first, second);

    return pymod(angle, (2 * PI));
}

Point rotate(Point o, Point p, float a) {
    float oX = o.x;
    float oY = o.y;
    float pX = p.x;
    float pY = p.y;
    a = -a;

    auto qX = oX + cos(a) * (pX - oX) - sin(a) * (pY - oY);
    auto qY = oY + sin(a) * (pX - oX) + cos(a) * (pY - oY);

    return Point(qX, qY);
}

float norm(float a, float b) {
    return sqrt(pow(a, 2) + pow(b, 2));
}

Slice!(ubyte*, 3, Contiguous) rotateImage(SliceKind kind)(Slice!(ubyte*, 3, kind) frame, float angle, Point center) {
    auto height = frame.shape[0];
    auto width = frame.shape[1];

    // OCV to DCV coordinate system conversion 
    center.x -= width / 2;
    center.y -= height / 2;
    auto transMat = transformation_matrix_2d(center, angle, 1.0);

    auto rotated = frame.dup.transformAffine(transMat);
    return rotated;
}

// taken from OpenCV formula, does not directly translate to dcv
Slice!(float*, 2, Contiguous) transformation_matrix_2d(Point center, float angle, float scale) {
    float alpha = scale * cos(angle);
    float beta = scale * sin(angle);

    return [
        [alpha, beta, (1 - alpha) * center.x - beta * center.y],
        [-beta, alpha, beta * center.x + (1 - alpha) * center.y],
        [0.0f, 0.0f, 1.0f],
    ].fuse;
}

bool intersects(BoundingBox b1, BoundingBox b2, float amount = 0.3) {
    auto area1 = b1.width * b1.height;
    auto area2 = b2.width * b2.height;
    float total = area1 + area2;

    float inter = 0.0;

    auto b1x2 = b1.point.x + b1.width;
    auto b1y2 = b1.point.y + b1.height;
    auto b2x2 = b2.point.x + b2.width;
    auto b2y2 = b2.point.y + b2.height;

    auto left = max(b1.point.x, b2.point.x);
    auto right = min(b1x2, b2x2);
    auto top = max(b1.point.y, b2.point.y);
    auto bottom = min(b1y2, b2y2);

    if (left < right && top < bottom) {
        inter = (right - left) * (bottom - top);
        total -= inter;
    }

    if (inter / total >= amount) {
        return true;
    }
    
    return false;
}

int[BoundingBox] groupRects(BoundingBox[] rects) {
    int[BoundingBox] groups;

    foreach (rect; rects) {
        groups[rect] = -1;
    }

    int groupId = 0;

    foreach (i, rect; rects) {
        auto group = groupId;
        groupId++;

        if (groups[rect] < 0) {
            groups[rect] = group;
        } else {
            group = groups[rect];
        }

        foreach (j, other; rects) {
            if (i == j) {
                continue;
            }
            if (rect.intersects(other)) {
                groups[other] = group;
            }
        }
    }

    return groups;
}
