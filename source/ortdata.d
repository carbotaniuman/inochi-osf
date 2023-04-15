module ortdata;

import bindbc.onnxruntime.config;
import bindbc.onnxruntime.v12.bind;
import bindbc.onnxruntime.v12.types;

import std.conv;

const(OrtApi)* ort;

shared static this() {
    const support = loadONNXRuntime();
    if (support == ONNXRuntimeSupport.noLibrary || support == ONNXRuntimeSupport.badLibrary)
    {
        throw new OrtException("No onnx library");
    }

    ort = OrtGetApiBase().GetApi(ORT_API_VERSION);
}

shared static ~this() {
    unloadONNXRuntime();
}

class OrtException : Exception
{
    this(string msg, string file = __FILE__, size_t line = __LINE__) {
        super(msg, file, line);
    }
}

void checkStatus(OrtStatus* status, string file = __FILE__, size_t line = __LINE__) {
    if (status) {
        auto ort = OrtGetApiBase().GetApi(ORT_API_VERSION);
        auto msg = ort.GetErrorMessage(status).to!string();
        ort.ReleaseStatus(status);
        throw new OrtException(msg, file, line);
    }
}