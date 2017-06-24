#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

using namespace std;
using namespace cv;
using namespace cv::ml;

#include "mousedraw.h"


#define resulttxt "/Users/aamirjawaid/Documents/workspace/hand-write-digit-recognition-with-opencv/t10k-images/result.txt"

static string modelFile = "/Users/aamirjawaid/Documents/workspace/hand-write-digit-recognition-with-opencv/ANN.xml";

void buildImg()
{
    BoxExtractor box;
    Mat res = box.MouseDraw ("draw",Mat(400,400,CV_8UC3,Scalar(0,0,0)),Scalar(255,255,255),18);
    imwrite ("/Users/aamirjawaid/Documents/workspace/hand-write-digit-recognition-with-opencv/test.jpg",res);
}

double getSkew(Mat& img) {
    Moments m = moments(img);
    if(abs(m.mu02) < 1e-2)
    {
        // No deskewing needed.
        return -1;
    }
    // Calculate skew based on central momemts.
    return m.mu11/m.mu02;
}

Mat deskew(Mat& img, double skew)
{
    Mat warpMat = (Mat_<double>(2,3) << 1, skew, -0.5*28*skew, 0, 1 , 0);
    
    Mat imgOut = Mat::zeros(img.rows, img.cols, img.type());
    warpAffine(img, imgOut, warpMat, imgOut.size(), WARP_INVERSE_MAP|INTER_LINEAR);
    
    return imgOut;
}

Mat deskew(Mat& img) {
    double skew = getSkew(img);
    if (skew == -1) {
        return img.clone();
    }
    
    return deskew(img, skew);
}

int getNumberOfLines(const string csvPath) {
    ifstream myfile(csvPath);
    int numberOfLines = 0;
    string line;
    if(myfile.is_open()){
        while(!myfile.eof()){
            std::getline(myfile, line);
            numberOfLines++;
        }
        myfile.close();
    }
    return --numberOfLines;
}

void getDescriptors(const string csvPath, Mat &dataMatrix, Mat &resultMatrix) {
    int numberOfLines = getNumberOfLines(csvPath);
    dataMatrix = Mat::zeros(numberOfLines, 324, CV_32FC1);
    resultMatrix = Mat::zeros(numberOfLines, 10, CV_32FC1);
    ifstream inputfile(csvPath);
    string current_line;
    
    int row = 0;
    getline(inputfile, current_line);
    while(getline(inputfile, current_line)){
        // Now inside each line we need to seperate the cols
        vector<uint8_t> values;
        stringstream temp(current_line);
        string single_value;
        int32_t result = -1;
        while(getline(temp,single_value,',')){
            if (result == -1) {
                result = atoi(single_value.c_str());
            } else {
                // convert the string element to a integer value
                values.push_back(atoi(single_value.c_str()));
            }
        }
        
        cv::Mat matrix = cv::Mat(values).reshape(1, 28);
        cv::Mat deskewedMatrix = deskew(matrix);
        HOGDescriptor *hog=new HOGDescriptor(cvSize(28,28),cvSize(14,14),cvSize(7,7),cvSize(7,7),9);
        vector<float> descriptors;
        hog->compute(deskewedMatrix, descriptors,Size(1,1), Size(0,0));
        int n=0;
        for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)
        {
            dataMatrix.at<float>(static_cast<int>(row), n) = *iter;
            n++;
        }
        resultMatrix.at<float>(static_cast<int>(row), result) = 1.f;
        row++;
        cout << row << endl;
    }
}

void getDescriptorsWith(const string imagePath, Mat* dataMatrixOutput, Mat* resultMatrixOutput) {
    vector<string> img_path;
    vector<int> img_catg;
    int nLine = 0;
    string buf;
    ifstream svm_data(imagePath);
    int n;
    while( svm_data )
    {
        if( getline( svm_data, buf ) )
        {
            nLine ++;
            if( nLine % 2 == 0 )
            {
                img_catg.push_back( atoi( buf.c_str() ) );
            }
            else
            {
                img_path.push_back( buf );
            }
        }
    }
    svm_data.close();
    int nImgNum = nLine / 2;
    Mat data_mat = Mat::zeros( nImgNum, 324, CV_32FC1 );
    Mat res_mat = Mat::zeros(nImgNum, 1, CV_32SC1);
    IplImage* src;
    IplImage* trainImg=cvCreateImage(cvSize(28,28),8,3);
    for( string::size_type i = 0; i != img_path.size(); i++ )
    {
        src=cvLoadImage(img_path[i].c_str(),1);
        if( src == NULL )
        {
            cout<<" can not load the image: "<<img_path[i].c_str()<<endl;
            continue;
        }
        cout<<"deal with\t"<<img_path[i].c_str()<<endl;
        cvResize(src,trainImg);
        HOGDescriptor *hog=new HOGDescriptor(cvSize(28,28),cvSize(14,14),cvSize(7,7),cvSize(7,7),9);
        vector<float>descriptors;
        
        Mat trainImg_ = cvarrToMat(trainImg);
        hog->compute(trainImg_, descriptors,Size(1,1), Size(0,0));
        cout<<"HOG dims: "<<descriptors.size()<<endl;
        n=0;
        for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)
        {
            data_mat.at<float>(static_cast<int>(i), n) = *iter;
            n++;
        }
        res_mat.at<int>(static_cast<int>(i), 0) = img_catg[i];
        cout<<"Done !!!: "<<img_path[i].c_str()<<" "<<img_catg[i]<<endl;
        cvReleaseImage(&src);
        cvReleaseImage(&trainImg);
    }
    
    dataMatrixOutput = &data_mat;
    resultMatrixOutput = &res_mat;
}

void testTrain(Mat data_mat, Mat res_mat, double C, double gamma)
{
    Ptr<ml::SVM> svm = ml::SVM::create();
    CvTermCriteria criteria;
    criteria = cvTermCriteria( CV_TERMCRIT_EPS, 100000, FLT_EPSILON );
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    svm->setC(C);
    svm->setGamma(gamma);
    svm->setTermCriteria(criteria);
    Ptr<ml::TrainData> tData = TrainData::create(data_mat, SampleTypes::ROW_SAMPLE, res_mat);
    svm->train(tData);
    std::cout<<"saving... ... !!! \n "<<endl;
    svm->save(modelFile);;
    cout << modelFile << "is saved !!! \n exit train process"<<endl;
}

void testTrain(Mat data_mat, Mat res_mat) {
    int networkInputSize = data_mat.cols;
    int networkOutputSize = res_mat.cols;
    cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::create();
    std::vector<int> layerSizes = { networkInputSize, data_mat.rows / (6 * (networkInputSize + networkOutputSize)), networkOutputSize };
    mlp->setLayerSizes(layerSizes);
    mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
    mlp->train(data_mat, cv::ml::ROW_SAMPLE, res_mat);
    std::cout<<"saving... ... !!! \n "<<endl;
    mlp->save(modelFile);;
    cout << modelFile << "is saved !!! \n exit train process"<<endl;
}

Mat getMatFromImage(const string imagePath) {
    IplImage *test;
    
    test = cvLoadImage("/Users/aamirjawaid/Documents/workspace/hand-write-digit-recognition-with-opencv/test.jpg", 1);
    cout<<"load image done"<<endl;
    IplImage* trainTempImg=cvCreateImage(cvSize(28,28),8,3);
    cvZero(trainTempImg);
    cvResize(test,trainTempImg);
    return cvarrToMat(trainTempImg);
}

int predict(Mat descriptor, Ptr<ml::SVM> svm) {
    return svm->predict(descriptor);
}

int testPredict(Mat img)
{
//    char result[300];
    Ptr<ml::SVM> svm = Algorithm::load<SVM>(modelFile);
    HOGDescriptor *hog=new HOGDescriptor(cvSize(28,28),cvSize(14,14),cvSize(7,7),cvSize(7,7),9);
    vector<float>descriptors;
    hog->compute(img, descriptors,Size(1,1), Size(0,0));
//    cout<<"HOG dims: "<<descriptors.size()<<endl;
    Mat SVMtrainMat = Mat::zeros(1, static_cast<int>(descriptors.size()), CV_32FC1);
    int n=0;
    for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)
    {
        SVMtrainMat.at<float>(0,n) = *iter;
        n++;
    }
    
    return svm->predict(SVMtrainMat);
//    sprintf(result, "%d\r\n",ret );
//    cvNamedWindow("test.jpg",1);
//    cvShowImage("test.jpg",test);
//    cout<<"predict result:"<<result<<endl;
//    waitKey ();
//    cvReleaseImage(&test);
//    cvReleaseImage(&trainTempImg);
}

void printMatrix(string filename, Mat matrix) {
    string folder = "/Users/aamirjawaid/Documents/workspace/hand-write-digit-recognition-with-opencv/images/";
    char fullfilepath[1024];
    sprintf(fullfilepath, "%s%s", folder.c_str(), filename.c_str());
    imwrite(filename, matrix);
}

void visualizeImages(const string csvPath) {
    ifstream inputfile(csvPath);
    string current_line;
    
    getline(inputfile, current_line);
    int numberOfSamples = 0;
    while(getline(inputfile, current_line) && numberOfSamples++ < 100){
        // Now inside each line we need to seperate the cols
        vector<uint8_t> values;
        stringstream temp(current_line);
        string single_value;
        int32_t result = -1;
        while(getline(temp,single_value,',')){
            if (result == -1) {
                result = atoi(single_value.c_str());
            } else {
                // convert the string element to a integer value
                values.push_back(atoi(single_value.c_str()));
            }
        }
        
        cv::Mat matrix = cv::Mat(values).reshape(1, 28);
        double skew = getSkew(matrix);
        cv::Mat deskewedMatrix = deskew(matrix, skew);
        cv::Mat deskewedMatrix2 = deskew(deskewedMatrix, skew);
        
        double skew2 = getSkew(deskewedMatrix2);
        cv::Mat deskewedMatrix3 = deskew(deskewedMatrix2, skew2);
        char filename[1024];
        sprintf(filename, "%d-e.jpg", numberOfSamples);
        printMatrix(filename, deskewedMatrix3);
    }
}

void test(Mat data, Mat result, int firstIndex = 0) {
    cv::Ptr<cv::ml::ANN_MLP> mlp = Algorithm::load<ANN_MLP>(modelFile);
    int numberOfWrongGuesses = 0;
    int numberOfRightGuesses = 0;
    for (int i = firstIndex; i < data.rows; i++) {
        cv::Mat output;
        mlp->predict(data.row(i), output);
        double max = -1;
        int index[2];
        minMaxIdx(output, NULL, &max, NULL, index);
        double max2 = -1;
        int index2[2];
        minMaxIdx(result.row(i), NULL, &max2, NULL, index2);
        cout << index[1] << " " << index2[1] << endl;
        if (index[1] == index2[1]) {
            numberOfRightGuesses++;
        } else {
            numberOfWrongGuesses++;
            cout << output << endl;
        }
    }
    
    cout << "numberOfRightGuesses";
    cout << numberOfRightGuesses << endl;

    cout << "numberOfWrongGuesses";
    cout << numberOfWrongGuesses << endl;
    
    float accuracy = static_cast<float>(numberOfRightGuesses) / (numberOfRightGuesses + numberOfWrongGuesses);
    cout << "Accuracy " << accuracy << endl;
}

void runThroughTests() {
    
    Mat allData;
    Mat res_mat;
    getDescriptors("/Users/aamirjawaid/Downloads/train.csv", allData, res_mat);
    
    
    int trainingDataCount = allData.rows * 0.80f;
    Mat trainingData;
    Mat trainResult;
    for (int i = 0; i < trainingDataCount; i++) {
        Mat row = allData.row(i);
        allData.row(i).copyTo(row);
        trainingData.push_back(row);
        
        Mat row2 = res_mat.row(i);
        res_mat.row(i).copyTo(row2);
        trainResult.push_back(row2);
    }
    
//    float highestAccuracy = 0.0;
//    double bestC = 0;
//    double bestGamma = 0;
    
//    BestC 10
//    Best Gamma0.5
    
//    double Gammas[] = {0.01, 0.05, 0.1, 0.5, 0.8};
//    double Cs[] = { 0.01, 0.1, 1, 10, 100 };
//    for (int j = 0; j < 5; j++) {
//        for (int k = 0; k < 5; k++ ) {
            testTrain(trainingData, trainResult);
//            cout << "Testing" << endl;
            test(allData, res_mat, trainingDataCount);
//            int numberOfWrongGuesses = 0;
//            int numberOfRightGuesses = 0;
//            Ptr<ml::SVM> svm = Algorithm::load<SVM>("/Users/aamirjawaid/Documents/workspace/hand-write-digit-recognition-with-opencv/HOG_SVM_DATA.xml");
//            for (int i = trainingDataCount; i < allData.rows; i++) {
//                int result = predict(allData.row(i), svm);
//                if (result == res_mat.at<int>(i)) {
//                    numberOfRightGuesses++;
//                } else {
//                    numberOfWrongGuesses++;
//                }
//            }
//            
//            cout << "numberOfRightGuesses";
//            cout << numberOfRightGuesses << endl;
//            
//            cout << "numberOfWrongGuesses";
//            cout << numberOfWrongGuesses << endl;
//            
//            float accuracy = static_cast<float>(numberOfRightGuesses) / (numberOfRightGuesses + numberOfWrongGuesses);
//            cout << accuracy << endl;
    
//            if (accuracy > highestAccuracy) {
//                highestAccuracy = accuracy;
//                bestC = Cs[k];
//                bestGamma = Gammas[j];
//            }
//        }
//    }
//    
//    cout << "BestC " << bestC << endl;
//    cout << "Best Gamma" << bestGamma << endl;
//    cout << "Best Accuracy" << highestAccuracy << endl;
}

int main()
{
    string input = "";
    int myNumber = 0;
    while (true) {
        cout << "Please Select: \n1  build the d:/test.jpg\n2  build the d:/HOG_SVM_DATA.xml\n3  predict the d:/test.jpg\n\n";
        getline(cin, input);
        stringstream myStream(input);
        if (myStream >> myNumber)
            break;
        cout << "Invalid number, please try again" << endl;
    }
    cout << "You entered: " << myNumber << endl << endl;
    switch( myNumber ){
    // update the selected bounding box
    case 1:
        buildImg(); // 1. build the d:/test.jpg
        break;
    case 2:
            runThroughTests();
//        testTrain();   // 2. build the d:/HOG_SVM_DATA.xml
        break;
    case 3:
//        testPredict();  // 3. predict the d:/dtest.jpg
        break;
    case 4:
            visualizeImages("/Users/aamirjawaid/Downloads/train.csv");
    }
    cout << endl;
    return 0;
}





