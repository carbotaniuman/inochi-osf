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

void main()
{
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
