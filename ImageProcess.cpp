#define _CRT_SECURE_NO_WARNINGS
#include<opencv2\opencv.hpp>
#include<iostream>
#include<fstream>
#include<corecrt_io.h>
#include<string>
#include<stdlib.h>
#include <stdio.h>
#include<fstream>
#include<string>

using namespace cv;

struct compareArea
{
	bool operator()(const std::vector<cv::Point>outcurves1, const std::vector<cv::Point>outCureves2)
	{
		return std::abs(contourArea(outcurves1)) >abs( contourArea(outCureves2));
	}
};

struct Compare
{
	bool operator()(const cv::Point &lhs, const cv::Point &rhs)
	{
		return (std::pow(lhs.x,2)+std::pow(rhs.y,2))<(std::pow(rhs.x, 2) + std::pow(rhs.y, 2));
	}
};

void SearchImage(std::string ImageFile, std::vector<std::string>&ImageList)
{
	std::string tempFile = ImageFile;
	tempFile.append("\\*.jpg");

	intptr_t handle;
	_finddata_t findData;

	handle = _findfirst(tempFile.c_str(), &findData);
	if (handle == -1)
		return;

	do {
		if (findData.attrib & _A_SUBDIR)
		{
			if (strcmp(findData.name, ".") == 0 || strcmp(findData.name, "..") == 0)
				continue;
		}
		else
		{
			std::string fileName = ImageFile+("\\")+findData.name;

			ImageList.push_back(fileName);
		}
	} while (_findnext(handle, &findData) == 0);

	_findclose(handle);  //关闭搜索句柄
}

std::vector<cv::Point2f>detectPoints(std::vector<std::vector<cv::Point>>curves,std::vector<cv::Point>outCurve, cv::Mat image)
{
	std::vector<cv::Point2f> points_(4);

	//找到4个极点
	std::vector<float>distancesToZero;
	std::vector<float>distancesToRightDown;
	std::vector<float>distancesToRightUp;
	std::vector<float>distancesToLeftDown;
	outCurve.clear();
	for (int j = 0; j < curves.size(); j++)
	{
		outCurve = curves[j];
		for (int i = 0; i < outCurve.size(); i++)
		{
			float distToZero = std::pow(outCurve[i].x, 2) + std::pow(outCurve[i].y, 2);
			float distToRightDown = std::pow(outCurve[i].x - image.cols, 2) + std::pow(outCurve[i].y - image.rows, 2);
			float distToRightUp = std::pow(outCurve[i].x - image.cols, 2) + std::pow(outCurve[i].y - 0, 2);
			float distToLeftDown = std::pow(outCurve[i].x - 0, 2) + std::pow(outCurve[i].y - image.rows, 2);
			distancesToZero.push_back(distToZero);
			distancesToRightDown.push_back(distToRightDown);
			distancesToRightUp.push_back(distToRightUp);
			distancesToLeftDown.push_back(distToLeftDown);
		}
	}
	outCurve.clear();

	sort(distancesToZero.begin(), distancesToZero.end());
	sort(distancesToRightDown.begin(), distancesToRightDown.end());
	sort(distancesToRightUp.begin(), distancesToRightUp.end());
	sort(distancesToLeftDown.begin(), distancesToLeftDown.end());

	cv::Point pointToZeroMax;
	cv::Point pointToRightDownMax;
	cv::Point pointToRightUpMax;
	cv::Point pointToLeftDownMax;
	for (int j = 0; j < curves.size(); j++)
	{
		outCurve = curves[j];
		for (int i = 0; i < outCurve.size(); i++)
		{
			float distToZero = std::pow(outCurve[i].x, 2) + std::pow(outCurve[i].y, 2);
			float distToRightDown = std::pow(outCurve[i].x - image.cols, 2) + std::pow(outCurve[i].y - image.rows, 2);
			float distToRightUp = std::pow(outCurve[i].x - image.cols, 2) + std::pow(outCurve[i].y - 0, 2);
			float distToLeftDown = std::pow(outCurve[i].x - 0, 2) + std::pow(outCurve[i].y - image.rows, 2);

			if (distToZero == distancesToZero[distancesToZero.size() - 1])
				pointToZeroMax = outCurve[i];

			if (distToRightDown == distancesToRightDown[distancesToRightDown.size() - 1])
				pointToRightDownMax = outCurve[i];

			if (distToRightUp == distancesToRightUp[distancesToRightUp.size() - 1])
				pointToRightUpMax = outCurve[i];

			if (distToLeftDown == distancesToLeftDown[distancesToLeftDown.size() - 1])
				pointToLeftDownMax = outCurve[i];
		}
	}

	points_[0] = pointToRightDownMax;
	points_[1] = pointToLeftDownMax;
	points_[2] = pointToRightUpMax;
	points_[3] = pointToZeroMax;

	return points_;
}

void detectFindContours(cv::Mat image,std::vector<cv::Point>&outCurve,std::vector<std::vector<cv::Point>>&curves,std::vector<double>&contoursLength,
	std::vector<double>&areaNum, std::vector<std::vector<cv::Point>>&contours,bool dilate)
{
	if (outCurve.size() != 0)
		outCurve.clear();
	if (curves.size() != 0)
		curves.clear();
	if (contoursLength.size() != 0)
		contoursLength.clear();
	if (areaNum.size() != 0)
		areaNum.clear();
	if (contours.size() != 0)
		contours.clear();

	std::vector<cv::Vec4i>hierarchy;
	cv::findContours(image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	std::vector<double>area;
	for (int i = 0; i < contours.size(); i++){
		area.push_back(contourArea(contours[i])); //面积
	}
	
	for (int i = area.size() - 1; i >= 0; i--)
	{
		double length = arcLength(contours[i], true);
		if (length < 6000)
			continue;
		contoursLength.push_back(length);
		approxPolyDP(contours[i], outCurve, 0.02*length, true);
		double curveArea = abs(contourArea(outCurve, true));
		std::cout << curveArea << "\n";

		if (!dilate)
		{
			if (outCurve.size() % 4 == 0 && curveArea > 5000)
			{
				curves.push_back(outCurve);
				areaNum.push_back(curveArea);
			}
		}
		else
		{
			if (outCurve.size() >= 4 && curveArea >= 10000)
			{
				curves.push_back(outCurve);
				areaNum.push_back(curveArea);
			}
		}
	}

	sort(contoursLength.begin(), contoursLength.end());

}

void findDisMinus(std::vector<std::vector<cv::Point>>&curves, float& pointsMaxX, float& pointsMaxY)
{
	for (int m = 0; m < curves.size(); m++)
	{
		std::vector<cv::Point>outCurve = curves[m];
		for (int i = 0; i < outCurve.size() - 1; i++)
		{
			for (int j = i + 1; j < outCurve.size(); j++)
			{
				float x = std::abs(outCurve[i].x - outCurve[j].x);
				float y = std::abs(outCurve[i].y - outCurve[j].y);
				if (x > pointsMaxX)
					pointsMaxX = (int)x;
				if (y > pointsMaxY)
					pointsMaxY = (int)y;
			}
		}
	}
}

void method1()
{
	std::ofstream out("F:\\迅雷下载\\RedeceParems\\report.txt", std::ios::out || std::ios::app);
	std::string filename = "F:\\迅雷下载\\SourceImg";
	std::vector<std::string>ImageList;
	SearchImage(filename, ImageList);
	for (int i =2; i < ImageList.size(); i++)
	{
		int pos = 0;
		pos = ImageList[i].rfind("I");
		if (-1 == pos)
			continue;
		std::string name_= ImageList[i].substr(pos, 13);
		std::string imageName =ImageList[i]; 

		cv::Mat dilateImage;
		cv::Mat erodeImage;

		std::string name = imageName;
		cv::Mat imageGray;
		cv::Mat imageCanny;
		cv::Mat midImage;
		cv::Mat image = cv::imread(name);
		cv::Mat img;
		cv::Mat lineImage;
		cv::Mat lineFilterImage = image.clone();
		cv::cvtColor(image, imageGray, CV_BGR2GRAY);

		cv::Mat circleImage = image.clone();
		cv::medianBlur(imageGray, imageGray, 33);
		Canny(imageGray, imageCanny, 30, 80);

		std::vector<std::vector<cv::Point>>contours;
		std::vector<cv::Point>outCurve;
		std::vector<std::vector<cv::Point>>curves;

		std::vector<double>contoursLength;
		std::vector<double>areaNum;

		detectFindContours(imageCanny, outCurve,curves,contoursLength,areaNum, contours,false);

		//判断是否检测出具体的四个角点
		if (curves.size() >= 1)
		{
			//筛选出面积最大的点围成的多边形
			std::sort(areaNum.begin(), areaNum.end());
			std::vector<std::vector<cv::Point>>curves1;
			for (int m = 0; m < curves.size(); m++)
			{
				std::vector<cv::Point>outCurve1 = curves[m];
				double curveArea = abs(cv::contourArea(outCurve1, true));
				if (curveArea == areaNum[areaNum.size() - 1])
					curves1.push_back(curves[m]);
			}

			outCurve.clear();
			float pointsMaxX = 0.0;
			float pointsMaxY = 0.0;

			//找到距离差
			findDisMinus(curves1, pointsMaxX, pointsMaxY);
			std::vector<std::vector<cv::Point>>distanceXY;
			std::vector<cv::Point>distance;
			std::vector<cv::Point>distanceToRightDown;

			std::vector<cv::Point2f>points_ = detectPoints(curves1, outCurve, image);
			std::vector<cv::Point2f> points(4);
			points[0] = cv::Point(0, 0);
			points[1] = cv::Point(pointsMaxX, 0);
			points[2] = cv::Point(0, pointsMaxY);
			points[3] = cv::Point(pointsMaxX, pointsMaxY);

			cv::Mat rotateImg(pointsMaxX, pointsMaxY, CV_8UC3, cv::Scalar(0, 0, 0));
			cv::Mat M = getPerspectiveTransform(points_, points);
			cv::warpPerspective(circleImage, rotateImg, M, cv::Size(pointsMaxX, pointsMaxY));
			
			char writeName[256];
			sprintf(writeName, "F:\\迅雷下载\\RedeceParems\\%s", name_);
			cv::imwrite(writeName, rotateImg);
			out << "Method:面积最大的点围成的多边形: " <<  name_<<"\n";
		}		
		else
		{		
			//std::vector<Point2f>points_ = detectPoints(curves, outCurve, image);
			cv::Mat dilateImage1;
			cv::Mat elements = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
			dilate(imageCanny, dilateImage, elements, cv::Point(-1, -1));

			detectFindContours(dilateImage, outCurve, curves, contoursLength, areaNum, contours,true);
			//points_.clear();
			if (curves.size() >= 1)
			{
				//筛选出面积最大的点围成的多边形
				std::sort(areaNum.begin(), areaNum.end());
				std::vector<std::vector<cv::Point>>curves1;
				for (int m = 0; m < curves.size(); m++)
				{
					std::vector<cv::Point>outCurve1 = curves[m];
					double curveArea = abs(cv::contourArea(outCurve1, true));
					if (curveArea == areaNum[areaNum.size() - 1])
						curves1.push_back(curves[m]);
				}

				outCurve.clear();
				float pointsMaxX = 0.0;
				float pointsMaxY = 0.0;

				//找到距离差
				findDisMinus(curves1, pointsMaxX, pointsMaxY);

				//std::cout << pointsMaxX << "  " << pointsMaxY << "\n";
				std::vector<std::vector<cv::Point>>distanceXY;
				std::vector<cv::Point>distance;
				std::vector<cv::Point>distanceToRightDown;

				std::vector<cv::Point2f>points_ = detectPoints(curves1, outCurve, image);
				std::vector<cv::Point2f> points(4);
				points[0] = cv::Point(0, 0);
				points[1] = cv::Point(pointsMaxX, 0);
				points[2] = cv::Point(0, pointsMaxY);
				points[3] = cv::Point(pointsMaxX, pointsMaxY);

				cv::Mat rotateImg(pointsMaxX, pointsMaxY, CV_8UC3, cv::Scalar(0, 0, 0));
				cv::Mat M = getPerspectiveTransform(points_, points);
				cv::warpPerspective(circleImage, rotateImg, M, cv::Size(pointsMaxX, pointsMaxY));
				char writeName[256];
				sprintf(writeName, "F:\\迅雷下载\\RedeceParems\\%s", name_);
				cv::imwrite(writeName, rotateImg);
				out << "Method:面积最大的点围成的多边形: " << name_ << "\n";
			}
			else //找到周长最大的多边形
			{
				//当得到的角点中的数没有时，需要换别的方法来进行处理
				//找到角点中围成的多边形中周长最长的多边形
				for (int i = 0; i < contours.size(); i++)
				{
					double length = cv::arcLength(contours[i], true);
					cv::approxPolyDP(contours[i], outCurve, 0.02*length, true);
					if (length == contoursLength[contoursLength.size() - 1])
					{
						curves.push_back(outCurve);
						break;
					}
				}

				//outCurve.clear();
				float pointsMaxX = 0.0;
				float pointsMaxY = 0.0;
				//找到距离差
				findDisMinus(curves, pointsMaxX, pointsMaxY);

				std::vector<std::vector<cv::Point>>distanceXY;
				std::vector<cv::Point>distance;
				std::vector<cv::Point>distanceToRightDown;

				std::vector<cv::Point2f>points_ = detectPoints(curves, outCurve, image);

				/*for (int i = 0; i < points_.size(); i++)
				{
				circle(circleImage, points_[i], 5, Scalar(0, 0, 255), 3);
				}*/

				std::vector<cv::Point2f> points(4);
				points[0] = cv::Point(0, 0);
				points[1] = cv::Point(pointsMaxX, 0);
				points[2] = cv::Point(0, pointsMaxY);
				points[3] = cv::Point(pointsMaxX, pointsMaxY);

				cv::Mat rotateImg(pointsMaxX, pointsMaxY, CV_8UC3, cv::Scalar(0, 0, 0));
				cv::Mat M = getPerspectiveTransform(points_, points);
				cv::warpPerspective(circleImage, rotateImg, M, cv::Size(pointsMaxX, pointsMaxY));
				char writeName[256];
				sprintf(writeName, "F:\\迅雷下载\\RedeceParems\\%s", name_);
				cv::imwrite(writeName, rotateImg);
				out << "Method:面积最大的点围成的多边形: " << name_ << "\n";
			}
		}
	}
}

void sauvola(cv::Mat img, cv::Mat imgGray, double k, int kernel_width,std::string imgName,std::string grayImgName)
{
	cv::Mat tempImg = imgGray.clone();
	cv::Mat intergralImage, squareImage;

	cv::integral(imgGray, intergralImage, squareImage,CV_64F,CV_64F); //calculate integral Image and sqaure Image
	
	int xmin, ymin, xmax, ymax;
	double mean, stdDev, threshold;//mean,stand deviation and threshold
	double intergralPixelValue(0.0), squarePixelValue(0.0);
	double mainDiagonalIntergralPixel(0.0), counterDiagonalIntergralPixel(0.0);
	double mainDiagonalSquareIntergralPixel(0.0), counterDiagonalSquareIntergralPixel(0.0);
	for (int i = 0; i < imgGray.rows; i++)
	{
		for (int j = 0; j < imgGray.cols; j++)
		{
			xmin = std::max(0, j - kernel_width);
			ymin = std::max(0, i - kernel_width);
			xmax = std::min(imgGray.cols - 1, j + kernel_width);
			ymax = std::min(imgGray.rows - 1, i + kernel_width);
			double area = (xmax - xmin+1)*(ymax - ymin+1);
			if (area < 0) {
				printf("Error in the area :%lf", area);
				return;
			}
			if (xmin == 0 && ymin == 0) { //if pixel point in the begining
				intergralPixelValue = intergralImage.at<double>(ymax, xmax);
				squarePixelValue = squareImage.at<double>(ymax, xmax);
			}
			else if (xmin == 0 && ymin > 0) { //pixel point in the first col
				intergralPixelValue = intergralImage.at<double>(ymax, xmax) - intergralImage.at<double>(ymin-1, xmax);
				squarePixelValue = squareImage.at<double>(ymax, xmax) - squareImage.at<double>(ymin - 1, xmax);
			}
			else if (xmin > 0 && ymin == 0) { //pixel point in the first row
				intergralPixelValue = intergralImage.at<double>(ymax, xmax) - intergralImage.at<double>(ymax, xmin - 1);
				squarePixelValue = squareImage.at<double>(ymax, xmax) - squareImage.at<double>(ymax, xmin - 1);
			}
			else { //the rest pixel of the image
				mainDiagonalIntergralPixel = intergralImage.at<double>(ymax, xmax) + intergralImage.at<double>(ymin-1,xmin-1);
				counterDiagonalIntergralPixel = intergralImage.at<double>(ymin-1, xmax) + intergralImage.at<double>(ymax, xmin - 1);
				intergralPixelValue = mainDiagonalIntergralPixel - counterDiagonalIntergralPixel;

				mainDiagonalSquareIntergralPixel = squareImage.at<double>(ymax, xmax) + squareImage.at<double>(ymin - 1, xmin-1);
				counterDiagonalSquareIntergralPixel = squareImage.at<double>(ymin - 1, xmax) + squareImage.at<double>(ymax, xmin - 1);
				squarePixelValue = mainDiagonalSquareIntergralPixel - counterDiagonalSquareIntergralPixel;
			}
			//The mean and stand deviation with the neighbourhood of the center (i,j)
			mean = intergralPixelValue / area;
			stdDev = std::sqrt((squarePixelValue - intergralPixelValue*intergralPixelValue / area) / (area - 1));
			threshold = mean*(1 + k*((stdDev / 128) - 1));
			if (imgGray.at<uchar>(i, j) > threshold) {
				tempImg.at<uchar>(i, j) = 255;
				for (int m = 0; m < 3; m++){
					img.at<cv::Vec3b>(i, j)[m] = 255;
				}
			}	
			if (imgGray.at<uchar>(i, j) < threshold) {
				tempImg.at<uchar>(i, j) = 0;
			}
		}
	}
	char writeName[256];
	char writeName1[256];
	sprintf(writeName, "F:\\迅雷下载\\imageEnhance\\%s", imgName);
	sprintf(writeName1, "F:\\迅雷下载\\imageEnhance\\%s", grayImgName);
	cv::imwrite(writeName1, grayImgName);
	cv::imwrite(writeName, img);

}

void ImageEnhance()
{
	char fileName[256];
	sprintf(fileName, "F:\\迅雷下载\\RedeceParems");
	std::vector<std::string>ImageList;
	SearchImage(fileName, ImageList);
	for (int i = 0; i < ImageList.size(); i++)
	{		
		int pos = ImageList[i].rfind("I");
		if (-1 == pos)
			continue;
		std::string name_ = ImageList[i].substr(pos, 13);
		std::string nameGray = "gray_"+name_ ;
		std::string imageName = ImageList[i]; 
		cv::Mat img2 = cv::imread(imageName);
		cv::Mat img2Gray;
		cv::cvtColor(img2, img2Gray, CV_BGR2GRAY);
		double k = 0.05;
		int kernel_width = 500;
		double x = kernel_width / 2;
		sauvola(img2,img2Gray, k, x,name_, nameGray);

	}
}

void ImageReProcess()
{
	int upToDownPixel = 0, leftToRightPixel = 0, downToUpPixel = 0, rightToLeftPixel = 0;
	int upToDownI = 0, leftToRightJ = 0, downToUpI = 0, rightToLeftJ = 0;
	int upToDownJ = 0, leftToRightI = 0, downToUpJ = 0, rightToLeftI = 0;
	std::vector<int>I;
	std::vector<int>J;
	Point point1, point2, point3, point4;
	std::string imgName = "F:\\迅雷下载\\RedeceParems\\IMG_9495.JPG";
	cv::Mat img = cv::imread(imgName);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.rows; j++)
		{
			//up to down
			if ((i < 150 && j < img.cols) || (i>=img.rows-10 && j<img.cols) || (i<img.rows && j<20) || (i<img.rows && j>=img.rows-20)) {
				int pixel1 = (int)img.at<Vec3b>(i, j)[0];
				int pixel2 = (int)img.at<Vec3b>(i, j)[1];
				int pixel3 = (int)img.at<Vec3b>(i, j)[2];

				if (pixel1 > 70 && pixel2 > 70 && pixel3 > 70) {
					pixel1 = 0;
					pixel2 = 0;
					pixel3 = 0;
				}
				img.at<Vec3b>(i, j)[0] = pixel1;
				img.at<Vec3b>(i, j)[1] = pixel2;
				img.at<Vec3b>(i, j)[2] = pixel3;
			}
		}
	}
	imshow("1", img);
	cv::waitKey(0);

	//up to down
	/*for (int i = 0; i < 150; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int pixel1 = (int)img.at<Vec3b>(i, j)[0];
			int pixel2 = (int)img.at<Vec3b>(i, j)[1];
			int pixel3 = (int)img.at<Vec3b>(i, j)[2];

			if (pixel1 > 70 && pixel2 > 70 && pixel3 > 70) {
				pixel1 = 0;
				pixel2 = 0;
				pixel3 = 0;
			}
			img.at<Vec3b>(i, j)[0] = pixel1;
			img.at<Vec3b>(i, j)[1] = pixel2;
			img.at<Vec3b>(i, j)[2] = pixel3;
		}
	}*/
	/*imshow("img1", img);
	cv::waitKey(0);*/
	
	//down to up
/*	for (int i = img.rows - 1; i >= img.rows - 10; i--)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int pixel1 = (int)img.at<Vec3b>(i, j)[0];
			int pixel2 = (int)img.at<Vec3b>(i, j)[1];
			int pixel3 = (int)img.at<Vec3b>(i, j)[2];

			if (pixel1 > 70 && pixel2 > 70 && pixel3 > 70) {
				pixel1 = 0;
				pixel2 = 0;
				pixel3 = 0;
			}
			img.at<Vec3b>(i, j)[0] = pixel1;
			img.at<Vec3b>(i, j)[1] = pixel2;
			img.at<Vec3b>(i, j)[2] = pixel3;
		}
	*///}


	for (int size = 0; size < 183; size++)
	{
		//only calc the edge pixel
		for (int i = 0; i < 200; i++)
		{
			for (int j = 0; j < img.cols; j++)
			{
				int pixel1 = (int)img.at<Vec3b>(i, j)[0];
				int pixel2 = (int)img.at<Vec3b>(i, j)[1];
				int pixel3 = (int)img.at<Vec3b>(i, j)[2];

				/*if (pixel1 > 70 && pixel2 > 70 && pixel3 > 70) {
					pixel1 = 0;
					pixel2 = 0;
					pixel3 = 0;
				}*/

				if (pixel1 < 70 && pixel2 < 70 && pixel3 < 70) {
					upToDownPixel++;
				}
				if ((upToDownPixel > 130 * img.cols)) {
					if (pixel1 > 70 && pixel2 > 70 && pixel3 > 70) {
						upToDownI = i;
						upToDownJ = j;
						point1 = Point(upToDownI, upToDownJ);
						break;
					}
				}
			}
			if (upToDownI && upToDownJ) {
				break;
			}
		}
		I.push_back(upToDownI);
		J.push_back(upToDownJ);

		//down to up
		for (int i = img.rows - 1; i >= img.rows - 50; i--)
		{
			for (int j = 0; j < img.cols; j++)
			{
				int pixel1 = (int)img.at<Vec3b>(i, j)[0];
				int pixel2 = (int)img.at<Vec3b>(i, j)[1];
				int pixel3 = (int)img.at<Vec3b>(i, j)[2];

				if (pixel1 < 70 && pixel2 < 70 && pixel3 < 70) {
					downToUpPixel++;
				}
				if ((downToUpPixel > 30 * img.cols)) {
					if (pixel1 > 70 && pixel2 > 70 && pixel3 > 70) {
						downToUpI = i;
						downToUpJ = j;
						point2 = Point(downToUpI, downToUpJ);
						break;
					}
				}
			}
			if (downToUpI && downToUpJ) {
				break;
			}
		}
		I.push_back(downToUpI);
		J.push_back(downToUpJ);

		//right to left
		for (int i = 0; i < img.rows; i++)
		{
			for (int j = img.cols - 1; j >= img.cols - 50; j--)
			{
				int pixel1 = (int)img.at<Vec3b>(i, j)[0];
				int pixel2 = (int)img.at<Vec3b>(i, j)[1];
				int pixel3 = (int)img.at<Vec3b>(i, j)[2];

				if (pixel1 < 70 && pixel2 < 70 && pixel3 < 70) {
					rightToLeftPixel++;
				}
				if ((rightToLeftPixel > 30 * img.rows)) {
					if (pixel1 > 70 && pixel2 > 70 && pixel3 > 70) {
						rightToLeftI = i;
						rightToLeftJ = j;
						point3 = Point(rightToLeftI, rightToLeftJ);
						break;
					}
				}
			}
			if (rightToLeftI && rightToLeftJ) {
				break;
			}
		}
		I.push_back(rightToLeftI);
		J.push_back(rightToLeftJ);

		//left to right
		for (int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j < 50; j++)
			{
				int pixel1 = (int)img.at<Vec3b>(i, j)[0];
				int pixel2 = (int)img.at<Vec3b>(i, j)[1];
				int pixel3 = (int)img.at<Vec3b>(i, j)[2];

				if (pixel1 < 70 && pixel2 < 70 && pixel3 < 70) {
					leftToRightPixel++;
				}
				if ((leftToRightPixel > 20 * img.rows)) {
					if (pixel1 > 70 && pixel2 > 70 && pixel3 > 70) {
						leftToRightI = i;
						leftToRightJ = j;
						point4 = Point(leftToRightI, leftToRightJ);
						break;
					}
				}
			}
			if (leftToRightI && leftToRightJ) {
				break;
			}
		}
		I.push_back(leftToRightI);
		J.push_back(leftToRightJ);

		if (!upToDownI || !upToDownJ || !downToUpI || !downToUpJ || !leftToRightI || !leftToRightJ || !rightToLeftI || !rightToLeftJ) {
			J.clear();
			I.clear();
			continue;
		}

		//find the max and min (i,j)
		std::sort(I.begin(), I.end());
		std::sort(J.begin(), J.end());
		int minI = I[0];
		int maxI = I[I.size() - 1];
		int minJ = J[0];
		int maxJ = J[J.size() - 1];

		Rect area(minI, minJ, maxJ - minJ, maxI - minI);
		Mat newImg = Mat::zeros(maxJ - minJ, maxI - minI, img.type());
		newImg=img(Rect(minJ, minI, maxJ - minJ, maxI - minI));
	
		cv::imshow("new", newImg);
		cv::waitKey(0);
	}	
}

int main(int argc, char **argv)
{
	method1();
	//ImageReProcess();
	ImageEnhance();//图像增强

	return 0;
}