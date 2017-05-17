//
//  main.cpp
//  hello
//
//  Created by candicecsy on 2017/5/1.
//  Copyright © 2017年 candicecsy. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>

using namespace std;

#define NUM_NEI (4)
#define DIM (3)

int ***pixels;
int xMax, yMax, zMax;

clock_t startTime;
clock_t endTime;
double total;
int fileNo;

void zoom(float ratioX, float ratioY);
void translate(int offsetX, int offsetY);
float computeDistance(float x1, float y1, float x2, float y2);

int main(int argc, const char * argv[]) {
    /* Load original image matrix data */

    startTime = clock();

    ifstream infile;
    for (int i = 1; i < argc; i++) {
        cout << "Image: " << argv[i] << endl;
        fileNo = i;

        infile.open(argv[i]);
    
        infile >> xMax >> yMax >> zMax;
    
        pixels = new int**[zMax];
        for (int z = 0; z < zMax; z++) {
            pixels[z] = new int*[xMax];
            for (int x = 0; x < xMax; x++) {
                pixels[z][x] = new int[yMax];
            }
        }
    
        for (int z = 0; z < zMax; z++) {
            for (int x = 0; x < xMax; x++) {
                for (int y = 0; y < yMax; y++) {
                    infile >> pixels[z][x][y];
                }
            }
        }
    
        infile.close();

        endTime = clock();
        cout << "Time for reading the input file is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    
        /* Zooming 2x */
        zoom(2, 2);
    
        /* Zooming 0.5x */
        zoom(0.5, 0.5);
    
        /* Translation with offset 200 in x-axis and 100 in y-axis */
   
        translate(200, 100);
    
        /* Free buffer storage of input and output image matrix */
        for (int z = 0; z < zMax; z++) {
            for (int x = 0; x < xMax; x++) {
                delete[] pixels[z][x];
            }
            delete[] pixels[z];
        }
        delete pixels;

        cout << endl;
    }

    /* Report total time */
    cout << "Total time for picture transformation is: " << total << "s" << endl;
    
    return 0;
}

void zoom(float ratioX, float ratioY) {
    /* Maximum x, y, and z of output image */
    int xMaxOut, yMaxOut, zMaxOut;
    xMaxOut = int(xMax * ratioX + 0.5);
    yMaxOut = int(yMax * ratioY + 0.5);
    zMaxOut = zMax;
    
    /* Create buffer storage for output image */
    int ***pixelsOut;
    pixelsOut = new int**[zMaxOut];
    for (int z = 0; z < zMaxOut; z++) {
        pixelsOut[z] = new int*[xMaxOut];
        for (int x = 0; x < xMaxOut; x++) {
            pixelsOut[z][x] = new int[yMaxOut];
        }
    }
    
    startTime = clock();

    /* Coordinates of output image */
    int coordinateZ;
    
    /* The Four-Nearest Intepolation */
    for (int z = 0; z < zMaxOut; z++) {
        coordinateZ = z;
        
        for (int x = 0; x < xMaxOut; x++) {
            for (int y = 0; y < yMaxOut; y++) {
                float tempX = (float)x / ratioX;
                float tempY = (float)y / ratioY;
                
                if (fmod(tempX, 1) == 0 && fmod(tempY, 1) == 0) {
                    int coordinateX = tempX;
                    int coordinateY = tempY;
                    
                    if (coordinateX >= 0 && coordinateX < xMax
                        && coordinateY >= 0 && coordinateY < yMax) {
                        pixelsOut[z][x][y] = pixels[coordinateZ][coordinateX][coordinateY];
                    } else {
                        pixelsOut[z][x][y] = 255;
                    }
                } else {
                    int pixVal = 0;
                    float distance[NUM_NEI] = {0.0};
                    float totalDistance = 0.0;
                    int val[NUM_NEI];
                    
                    int coordinateX1 = floor((float)x / ratioX);
                    int coordinateY1 = floor((float)y / ratioY);
                    
                    distance[0] = computeDistance((float)coordinateX1, (float)coordinateY1, tempX, tempY);
                    totalDistance += distance[0];
                    if (coordinateX1 >= 0 && coordinateX1 < xMax
                        && coordinateY1 >= 0 && coordinateY1 < yMax) {
                        val[0] = pixels[coordinateZ][coordinateX1][coordinateY1];
                    } else {
                        val[0] = 255;
                    }
                    
                    
                    int coordinateX2 = coordinateX1;
                    int coordinateY2 = coordinateY1 + 1;
                    
                    distance[1] = computeDistance((float)coordinateX2, (float)coordinateY2, tempX, tempY);
                    totalDistance += distance[1];
                    if (coordinateX2 >= 0 && coordinateX2 < xMax
                        && coordinateY2 >= 0 && coordinateY2 < yMax) {
                        val[1] = pixels[coordinateZ][coordinateX2][coordinateY2];
                    } else {
                        val[1] = 255;
                    }
                    
                    int coordinateX3 = coordinateX1 + 1;
                    int coordinateY3 = coordinateY1;
                    
                    distance[2] = computeDistance((float)coordinateX3, (float)coordinateY3, tempX, tempY);
                    totalDistance += distance[2];
                    if (coordinateX3 >= 0 && coordinateX3 < xMax
                        && coordinateY3 >= 0 && coordinateY3 < yMax) {
                        val[2] = pixels[coordinateZ][coordinateX3][coordinateY3];
                    } else {
                        val[2] = 255;
                    }
                    
                    int coordinateX4 = coordinateX1 + 1;
                    int coordinateY4 = coordinateY1 + 1;
                    
                    distance[3] = computeDistance((float)coordinateX4, (float)coordinateY4, tempX, tempY);
                    totalDistance += distance[3];
                    if (coordinateX4 >= 0 && coordinateX4 < xMax
                        && coordinateY4 >= 0 && coordinateY4 < yMax) {
                        val[3] = pixels[coordinateZ][coordinateX3][coordinateY3];
                    } else {
                        val[3] = 255;
                    }
                    
                    for (int i = 0; i < NUM_NEI; i++) {
                        pixVal += distance[i] / totalDistance * val[i];
                    }
                    
                    pixelsOut[z][x][y] = pixVal;
                }
                
            }
        }
    }

    /* Report time for zooming the input picture */
    endTime = clock();
    total += (double)(endTime - startTime) / CLOCKS_PER_SEC;
    cout << "Time for zooming the input picture with " + to_string(ratioX) + "x is: "
            << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    
    /* Write output image matrix into file */
    ofstream outfile;
    outfile.open("output/image" + to_string(fileNo) + "/zoom_ratio" + to_string(ratioX) + ".txt");
    
    outfile << xMaxOut << " " << yMaxOut << " " << zMaxOut << endl;
    for (int z = 0; z < zMaxOut; z++) {
        for (int x = 0; x < xMaxOut; x++) {
            for (int y = 0; y < yMaxOut; y++) {
                outfile << pixelsOut[z][x][y] << " ";
            }
            outfile << endl;
        }
        outfile << endl;
    }
    outfile.close();
    
    for (int z = 0; z < zMaxOut; z++) {
        for (int x = 0; x < xMaxOut; x++) {
            delete[] pixelsOut[z][x];
        }
        delete[] pixelsOut[z];
    }
    delete pixelsOut;
}

void translate(int offsetX, int offsetY) {
    /* Maximum x, y, and z of output image */
    int xMaxOut, yMaxOut, zMaxOut;
    xMaxOut = xMax;
    yMaxOut = yMax;
    zMaxOut = zMax;
    
    /* Create buffer storage for output image */
    int ***pixelsOut;
    pixelsOut = new int**[zMaxOut];
    for (int z = 0; z < zMaxOut; z++) {
        pixelsOut[z] = new int*[xMaxOut];
        for (int x = 0; x < xMaxOut; x++) {
            pixelsOut[z][x] = new int[yMaxOut];
        }
    }

    startTime = clock();

    /* Coordinates of output image */
    int coordinateZ;
    
    /* The Four-Nearest Intepolation */
    for (int z = 0; z < zMaxOut; z++) {
        coordinateZ = z;
        
        for (int x = 0; x < xMaxOut; x++) {
            for (int y = 0; y < yMaxOut; y++) {
                int coordinateX = x - offsetX;
                int coordinateY = y - offsetY;
                
                if (coordinateX >= 0 && coordinateX < xMax
                    && coordinateY >= 0 && coordinateY < yMax) {
                    pixelsOut[z][x][y] = pixels[coordinateZ][coordinateX][coordinateY];
                } else {
                    pixelsOut[z][x][y] = 255;
                }
            }
        }
    }

    /* Report time for translating the input picture */
    endTime = clock();
    total += (double)(endTime - startTime) / CLOCKS_PER_SEC;
    cout << "Time for translating the input picture with x-axis offset 200 and y-axis offset 100 is: "
            << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    
    /* Write output image matrix into file */
    ofstream outfile;
    outfile.open("output/image" + to_string(fileNo) + "/translation.txt");
    
    outfile << xMaxOut << " " << yMaxOut << " " << zMaxOut << endl;
    for (int z = 0; z < zMaxOut; z++) {
        for (int x = 0; x < xMaxOut; x++) {
            for (int y = 0; y < yMaxOut; y++) {
                outfile << pixelsOut[z][x][y] << " ";
            }
            outfile << endl;
        }
        outfile << endl;
    }
    outfile.close();
    
    for (int z = 0; z < zMaxOut; z++) {
        for (int x = 0; x < xMaxOut; x++) {
            delete[] pixelsOut[z][x];
        }
        delete[] pixelsOut[z];
    }
    delete pixelsOut;
}

float computeDistance(float x1, float y1, float x2, float y2) {
    return pow((x1 - x2), 2) + pow((y1 - y2), 2);
}
