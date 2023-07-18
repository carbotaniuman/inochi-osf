import faceinfo;
import tracker;
import retinaface;

import std.conv;
import std.stdio;
import std.typecons : tuple;

import dcv.core;
import dcv.imgproc;
import dcv.imageio;
import dcv.plot;

import inmath;

void main()
{
    // auto aa = quat(-0.4292, 0.64783, 0.62051, 0.10518);
    // writeln(aa.normalized);

    // auto axis = aa.toAxisAngle;
    // writeln(axis);
    // writeln(quat.axisRotation(axis.length, axis.normalized).normalized);
    
    FaceData[] faces = [];
    Image image = imread("a.jpg");

    auto a = new Tracker(1280, 720);
    a.predict(image);

    // faces ~= a.detectFaces(image);
    
    // auto rf = new RetinaFace(4, 0.2);
    // faces ~= rf.detectFaces(image);

    // writeln(faces);

    // auto f = figure("Camera");
    // f.draw(image);
    // f.show();

    // while (waitKey(100uL) != 'q') {
    //     foreach (face; faces) {
    //         f.drawLine(PlotPoint(face.x, face.y), PlotPoint(face.x, face.y + face.height), plotRed, 1.0);
    //         f.drawLine(PlotPoint(face.x, face.y), PlotPoint(face.x + face.width, face.y), plotRed, 1.0);
    //         f.drawLine(PlotPoint(face.x + face.width, face.y), PlotPoint(face.x + face.width, face.y + face.height), plotRed, 1.0);
    //         f.drawLine(PlotPoint(face.x, face.y + face.height), PlotPoint(face.x + face.width, face.y + face.height), plotRed, 1.0);
    //     }
    // }
}
