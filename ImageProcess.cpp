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

void commentOut()
{
	//再以pointToZeroMax和pointToRightDownMax为中心，在某个区间内的点不要，过滤掉一些点
	//std::vector<Point2f>disPointsFilter;
	//for (int i = 0; i < outCurve.size(); i++)
	//{
	//	float disToPointToZeroMax = std::sqrt(std::pow(outCurve[i].x - pointToZeroMax.x, 2) + std::pow(outCurve[i].y - pointToZeroMax.y, 2));
	//	float distToPointToRightDownMax = std::sqrt(std::pow(outCurve[i].x - pointToRightDownMax.x, 2) + std::pow(outCurve[i].y - pointToRightDownMax.y, 2));
	//	if ((disToPointToZeroMax < 1000 && disToPointToZeroMax!=0) || (distToPointToRightDownMax < 1000 && distToPointToRightDownMax!=0))
	//		continue;
	//	disPointsFilter.push_back(outCurve[i]);
	//}

	//Point pointXY;
	//float distToZeroMax = std::pow(distance[0].x, 2) + std::pow(distance[0].y, 2);
	//for (int i = 1; i < distance.size(); i++)
	//{
	//	float dis = std::pow(distance[i].x, 2) + std::pow(distance[i].y, 2);
	//	if (distToZeroMax < dis)
	//		distToZeroMax = dis;
	//}
	//for (int i = 0; i < distance.size(); i++)
	//{
	//	float dis = std::pow(distance[i].x, 2) + std::pow(distance[i].y, 2);
	//	if (dis == distToZeroMax)
	//		pointXY = distance[i];
	//}

	//for (int i = 0; i < outCurve.size(); i++)
	//{
	//	circle(circleImage, outCurve[i], 5, Scalar(0, 0, 255), 3);
	//}

	//resize(circleImage, circleImage, Size(circleImage.rows / 5, circleImage.rows / 5));
	//imshow("circle", circleImage);
	//cv::waitKey(0);

	//points_[3] = pointXY;//找到离原点最远的点作为 points_[3]
	//for (auto it = distance.begin(); it != distance.end(); it++)
	//{
	//	if (*it == pointXY)
	//	{
	//		distance.erase(it);
	//		break;
	//	}
	//}

	////依次判断检测出的点与points_[3]的距离，最远的点做为points_[0]
	//std::vector<float>piToDis;
	//for (int i = 0; i < distance.size(); i++)
	//{
	//	float dist = std::pow(distance[i].x - points_[3].x, 2) + std::pow(distance[i].y - points_[3].y, 2);
	//	if (dist < 1000)
	//		continue;
	//	piToDis.push_back(dist);
	//}
	//sort(piToDis.begin(), piToDis.end());			
	//for (size_t i = 0; i < distance.size(); i++)
	//{
	//	if ((std::pow(distance[i].x - points_[3].x, 2) + std::pow(distance[i].y - points_[3].y, 2)) < 1000)
	//		continue;

	//	if (std::pow(distance[i].x - points_[3].x, 2) + std::pow(distance[i].y - points_[3].y, 2) == piToDis[0])
	//		points_[2] = distance[i];
	//	if (std::pow(distance[i].x - points_[3].x, 2) + std::pow(distance[i].y - points_[3].y, 2) == piToDis[1])
	//		points_[1] = distance[i];
	//	if (std::pow(distance[i].x - points_[3].x, 2) + std::pow(distance[i].y - points_[3].y, 2) == piToDis[piToDis.size()-1])
	//		points_[0] = distance[i];
	//}

	//std::vector<Vec4i>lines;
	//threshold(imageCanny, imageCanny, 200, 255, THRESH_BINARY);
	//HoughLinesP(imageCanny, lines, 1, CV_PI / 180, 80, 400, 100);
	//std::cout << "Lines before: " << lines.size() << "\n";
	//for (int i = 0; i < lines.size(); i++)
	//{
	//	line(image, Point2f(lines[i][0], lines[i][1]), Point2f(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 3);			
	//}		
	////resize(image, lineImage, Size(image.rows / 5, image.rows / 5));
	////imshow("lineImage", lineImage);
	////cv::waitKey(0);
	///*int maxLinesNum = 10;
	//int cannyThreshold = 80;
	//float factor = 2.5;
	//while (lines.size() >= 20)
	//{
	//	cannyThreshold += 2;
	//	Canny(imageGray, midImage, cannyThreshold, cannyThreshold*factor);
	//	threshold(midImage, midImage, 128, 255, THRESH_BINARY);
	//	HoughLinesP(midImage, lines, 1, CV_PI / 180, 80, 100, 100);
	//}
	//std::cout << "cannyThreshold" << cannyThreshold << "\n";
	//Canny(imageGray, midImage, cannyThreshold, cannyThreshold*factor);*/
	////resize(image, image, Size(image.rows / 5, image.cols / 5));
	////imshow("afterFilter", image);
	////threshold(midImage, midImage, 128, 255, THRESH_BINARY);
	////resize(midImage, resizeImage_2, Size(image.rows / 5, image.cols / 5));
	////imshow("afterThreshold", resizeImage_2);

	///*HoughLinesP(midImage, lines, 1, CV_PI / 180, 80, 100, 100);
	//for (int i = 0; i < lines.size(); i++)
	//{
	//	line(lineFilterImage, Point2f(lines[i][0], lines[i][1]), Point2f(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 3);
	//}
	//resize(lineFilterImage, lineFilterImage, Size(lineFilterImage.rows / 5, lineFilterImage.rows / 5));
	//imshow("lineFilterImage", lineFilterImage);
	//waitKey(0);

	//std::cout << "Lines after: " << lines.size() << "\n";*/
	//
	//std::vector<Point2f>crossPoints=getCrossPoint(lines);

	////确定旋转后的图像的大小
	//float pointsMaxX = 0.0;
	//float pointsMaxY = 0.0;
	//for (int i = 0; i < crossPoints.size(); i++)
	//{								
	//	for (int j = i + 1; j < crossPoints.size(); j++)
	//	{
	//		float x = std::abs(crossPoints[i].x - crossPoints[j].x);
	//		float y = std::abs(crossPoints[i].y - crossPoints[j].y);
	//		if (x > pointsMaxX)
	//			pointsMaxX = (int)x;
	//		if (y > pointsMaxY)
	//			pointsMaxY = (int)y;
	//	}				
	//}
	////std::cout << pointsMaxX << "  " << pointsMaxY << "\n";

	//for (int i = 0; i < crossPoints.size(); i++)
	//{
	//	for (int j = i + 1; j < crossPoints.size(); j++)
	//	{
	//		float x = std::abs(crossPoints[i].x - crossPoints[j].x);
	//		float y = std::abs(crossPoints[i].y - crossPoints[j].y);
	//		if (int(x) == pointsMaxX)
	//		{
	//			std::cout << crossPoints[i].x << "   " << crossPoints[i].y << "\n";
	//			std::cout << crossPoints[j].x << "   " << crossPoints[j].y << "\n";
	//		}
	//		std::cout << "\n";
	//		if (int(y) == pointsMaxY)
	//		{
	//			std::cout << crossPoints[i].x << "   " << crossPoints[i].y << "\n";
	//			std::cout << crossPoints[j].x << "   " << crossPoints[j].y << "\n";
	//		}
	//	}
	//}

	////找到离图像原点和图像右下角点最远的两个点
	//std::vector<float>distancesToZero;
	//std::vector<float>distancesToRightDown;
	//for (int i = 0; i < crossPoints.size(); i++)
	//{
	//	float distToZero = std::pow(crossPoints[i].x, 2) + std::pow(crossPoints[i].y, 2);
	//	float distToRightDown = std::pow(crossPoints[i].x - image.cols, 2) + std::pow(crossPoints[i].y - image.rows, 2);
	//	distancesToZero.push_back(distToZero);
	//	distancesToRightDown.push_back(distToRightDown);
	//}
	//sort(distancesToZero.begin(), distancesToZero.end());
	//sort(distancesToRightDown.begin(), distancesToRightDown.end());

	//Point pointToZeroMax;
	//Point pointToRightDownMax;

	//for (int i = 0; i < crossPoints.size(); i++)
	//{
	//	float distToZero = std::pow(crossPoints[i].x, 2) + std::pow(crossPoints[i].y, 2);
	//	float distToRightDown = std::pow(crossPoints[i].x - image.cols, 2) + std::pow(crossPoints[i].y - image.rows, 2);
	//	if (distToZero == distancesToZero[distancesToZero.size() - 1])
	//		pointToZeroMax = crossPoints[i];

	//	if (distToRightDown == distancesToRightDown[distancesToRightDown.size() - 1])
	//		pointToRightDownMax = crossPoints[i];
	//}

	////再以pointToZeroMax和pointToRightDownMax为中心，在某个区间内的点不要，过滤掉一些点
	//std::vector<Point2f>crossPointsFilter;
	//for (int i = 0; i < crossPoints.size(); i++)
	//{
	//	float disToPointToZeroMax = std::sqrt(std::pow(crossPoints[i].x - pointToZeroMax.x, 2) + std::pow(crossPoints[i].y - pointToZeroMax.y, 2));
	//	float distToPointToRightDownMax = std::sqrt(std::pow(crossPoints[i].x - pointToRightDownMax.x, 2) + std::pow(crossPoints[i].y - pointToRightDownMax.y, 2));
	//	if (disToPointToZeroMax < 1000 || distToPointToRightDownMax < 1000)
	//		continue;
	//	crossPointsFilter.push_back(crossPoints[i]);
	//}
}

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

std::vector<cv::Point>pointsFilter(std::vector<cv::Point>&points)
{
	std::vector<cv::Point>candidates(points);
	std::vector<cv::Point>filter(points);
	for (auto &i = candidates.begin(); i != candidates.end();)
	{
		for (auto &j = filter.begin(); j != filter.end();j++)
		{
			if (abs((*i).x - (*j).x) < 5 && abs((*i).y - (*j).y) < 5 && abs((*i).x - (*j).x) > 0 && abs((*i).y - (*j).y) > 0)
				i = filter.erase(i);
			else
				i++;
		}
	}

	return filter;
}

std::vector<cv::Point2f> getCrossPoint(std::vector<cv::Vec4i>lines)
{
	std::vector<cv::Point2f>crossPointsVector;
	cv::Point2f corssPoints;
	
	for (int i = 0; i < lines.size() - 1; i++)
	{
		double dy1 = lines[i][3] - lines[i][1];
		double dx1 = lines[i][2] - lines[i][0];
		
		for (int j = i + 1; j < lines.size(); j++)
		{
			double dy2 = lines[j][3] - lines[j][1];
			double dx2 = lines[j][2] - lines[j][0];

			if (dy1*dx2 == dx1*dy2)
				continue;
			else
			{
				double line2xMinusLine1_X = lines[j][0] - lines[i][0];
				double line2xMinusLine1_Y = lines[j][1] - lines[i][1];

				int A = dx1*lines[i][1] - dy1*lines[i][0];
				int B = dx2*lines[j][1] - dy2*lines[j][0];

				corssPoints.x = (dx1*dx2*line2xMinusLine1_Y + dy1*dx2*lines[i][0] - dy2*dx1*lines[j][0]) / (dy1*dx2 - dx1*dy2);
				corssPoints.y= -(dy1*dy2*line2xMinusLine1_X + dx1*dy2*lines[i][1] - dx2*dy1*lines[j][1]) / (dy1*dx2 - dx1*dy2);

				if (std::abs(corssPoints.x) > 3000	|| std::abs(corssPoints.y)>4000 || corssPoints.x<100 || corssPoints.y<100 )
					continue;

				crossPointsVector.push_back(corssPoints);
			}		
		}
	}

	for (int i = 0; i < crossPointsVector.size(); i++)
	{
		cv::Point2f points = crossPointsVector[i];

		std::cout << points.x << "  " << points.y<<"\n";
	}

	return crossPointsVector;
}

void DoEdgeDetect(cv::Mat image/*,std::string str*/)
{
	cv::Mat img = image.clone();
	cv::Mat contourImg = image.clone();
	cv::Mat pointsImage = image.clone();
	cv::Mat resizeImgae;
	cv::Mat resizeMidImg;
	cv::Mat resizeCannyImg;
	cv::Mat resizeImage_1;
	cv::Mat resizeImage_2;
	cv::Mat resizeImage_3;
	cv::Mat resizeImage_4;
	cv::Mat img1 = image.clone();
	cv::RNG rng(time(0));
	cv::Mat rotateImage = image.clone();
	cv::Mat imageGray;
	cvtColor(image, imageGray, CV_BGR2GRAY);
	GaussianBlur(imageGray, imageGray, cv::Size(3, 3), 0, 0);
	resize(imageGray, resizeImage_4, cv::Size(image.rows / 5, image.cols / 5));
	imshow("gray", resizeImage_4);
	
	int cannyThreshold = 80;
	float factor = 2.5;
	const int maxLinesNum = 10;//最多检测出的直线数
	cv::Mat midImage;
	Canny(imageGray, midImage, cannyThreshold, cannyThreshold*factor);
	resize(midImage, resizeMidImg, cv::Size(image.rows / 5, image.cols / 5));
	imshow("cannyImage", resizeMidImg);
	//std::string writeName = "F:\\迅雷下载\\TargetImage\\" + str;
	//imwrite(writeName, midImage);

	cv::threshold(midImage, midImage, 128, 255, cv::THRESH_BINARY);
	resize(midImage, resizeCannyImg, cv::Size(image.rows / 5, image.cols / 5));
	imshow("thresholdImage", resizeCannyImg);
		
	std::vector<cv::Vec4i>lines;
	HoughLinesP(midImage, lines, 1, CV_PI / 180, 80, 100, 100);
	std::cout << "Lines before: " << lines.size() << "\n";

	for (int i = 0; i < lines.size(); i++)
	{
		line(image, cv::Point2f(lines[i][0], lines[i][1]), cv::Point2f(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 255), 3);
	}
	resize(image, resizeImgae, cv::Size(image.rows / 5, image.cols / 5));
	imshow("linesImgBefore", resizeImgae);
	cv::waitKey(0);

	while (lines.size()>= maxLinesNum)
	{
		cannyThreshold += 2;
		Canny(imageGray, midImage, cannyThreshold, cannyThreshold*factor);
		threshold(midImage, midImage, 128, 255, cv::THRESH_BINARY);
		HoughLinesP(midImage, lines, 1, CV_PI / 180, 80, 100, 100);
	}
	std::cout << "cannyThreshold1: " << cannyThreshold << "\n";
	
	Canny(imageGray, midImage, cannyThreshold, cannyThreshold*factor);
	resize(image, resizeImage_1, cv::Size(image.rows / 5, image.cols / 5));
	imshow("afterFilter", resizeImage_1);

	threshold(midImage, midImage, 128, 255, cv::THRESH_BINARY);
	resize(midImage, resizeImage_2, cv::Size(image.rows / 5, image.cols / 5));
	imshow("afterThreshold", resizeImage_2);
	
	HoughLinesP(midImage, lines, 1, CV_PI / 180, 80, 100, 100);
	std::cout << "Lines after: " << lines.size() << "\n";
	for (int i = 0; i < lines.size(); i++)
	{
		line(img, cv::Point2f(lines[i][0], lines[i][1]), cv::Point2f(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 255), 3);
		//std::cout << sqrt(std::pow(lines[i][0] - lines[i][2], 2) + std::pow(lines[i][1] - lines[i][3], 2)) << "\n";
		//imshow("lineImages", img);
		//waitKey(0);
	}
	resize(img, resizeImage_3, cv::Size(image.rows / 5, image.cols / 5));
	imshow("linesImgAfterFilter", resizeImage_3);
	cv::waitKey(0);

	std::vector<cv::Vec4i>filterLines;
	for (int i = 0; i < lines.size(); i++)
	{
		if (std::abs(lines[i][0] - lines[i][2])<200 && std::abs(lines[i][1] - lines[i][3])<200)
			continue;
		filterLines.push_back(lines[i]);
	}
	std::cout << "filterLines" << filterLines.size() << "\n";

	std::vector<cv::Point2f> crossPoints = getCrossPoint(filterLines);

	std::cout << "交点个数：" << crossPoints.size() << "\n";
	for (int i = 0; i < crossPoints.size(); i++)
	{
		std::cout << crossPoints[i].x << "   " << crossPoints[i].y << "\n";
		circle(contourImg, crossPoints[i], 5, cv::Scalar(0, 0, 255));
		imshow("contourImg", contourImg);
		cv::waitKey(0);
	}

	//找到交点最大和最小的x,y值
	/*float MaxX = 0.0,float MaxY = 0.;
	float MinX = 0., float MinY = 0.;*/
	float pointsMaxX = crossPoints[0].x;
	float pointsMaxY = crossPoints[0].y;
	float pointsMinx = crossPoints[0].x;
	float pointsMinY = crossPoints[0].y;
	
	for (int i = 1; i < crossPoints.size(); i++)
	{
		if (crossPoints[i].x > pointsMaxX)
			pointsMaxX = crossPoints[i].x;
		if (crossPoints[i].y > pointsMaxY)
			pointsMaxY = crossPoints[i].y;
		if (crossPoints[i].x < pointsMinx)
			pointsMinx = crossPoints[i].x;
		if (crossPoints[i].y < pointsMinY)
			pointsMinY = crossPoints[i].y;
	}
	std::cout << "MaxX :" << pointsMaxX << "\n";
	std::cout << "MaxY :" << pointsMaxY << "\n";
	std::cout << "MinX :" << pointsMinx << "\n";
	std::cout << "MinY :" << pointsMinY << "\n";

	circle(img1, cv::Point2f(pointsMinx, pointsMinY), 2, cv::Scalar(0, 0, 255));
	circle(img1, cv::Point2f(pointsMinx, pointsMaxY), 2, cv::Scalar(0, 0, 255));
	circle(img1, cv::Point2f(pointsMaxX, pointsMinY), 2, cv::Scalar(0, 0, 255));
	circle(img1, cv::Point2f(pointsMaxX, pointsMaxY), 2, cv::Scalar(0, 0, 255));
	imshow("img1_1", img1);
	cv::waitKey(0);
	
	////旋转之后的点
	//for (int i = 0; i < crossPoints.size() - 1; i++)
	//{
	//	for (int j = i + 1; j < crossPoints.size(); j++)
	//	{
	//		float x = std::abs(crossPoints[i].x - crossPoints[j].x);
	//		float y = std::abs(crossPoints[i].y - crossPoints[j].y);
	//		if (x > pointsMaxX)
	//			pointsMaxX = (int)x;
	//		if (y > pointsMaxY)
	//			pointsMaxY = (int)y;
	//	}
	//}
	//std::vector<Point2f> points;
	//points.push_back(Point2f(0, 0));
	//points.push_back(Point2f(pointsMaxX, 0));
	//points.push_back(Point2f(0, pointsMaxY));
	//points.push_back(Point2f(pointsMaxX, pointsMaxY));

	//Mat rotateImg(pointsMaxX, pointsMaxY,CV_8UC3,Scalar(0,0,0));
	//Mat M= getPerspectiveTransform(crossPoints, points);
	//warpPerspective(rotateImage, rotateImg, M, Size(pointsMaxX, pointsMaxY));
	//imshow("rotateImg", rotateImage);
	//waitKey(0);
	
	std::cout << "\n";
	std::cout << "afterFilter :" << filterLines.size() << "\n";
	for (int i = 0; i < filterLines.size(); i++)
	{
		line(img1, cv::Point2f(filterLines[i][0], filterLines[i][1]), cv::Point2f(filterLines[i][2], filterLines[i][3]), cv::Scalar(0, 0, 255), 2);
		imshow("afterFilterImg", img1);
		cv::waitKey(0);
	}
	
	std::vector<std::vector<cv::Point>>contours;
	std::vector<cv::Vec4i>hierarchy;
	findContours(midImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	std::cout << "contoursSize: " << contours.size() << "\n";
	
	std::vector<std::vector<cv::Point>> contours_poly(contours.size());//用于存放折线点集
	for (int i = 0; i<contours.size(); i++)
	{
		cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 15, true);
		//drawContours(contourImg, contours_poly, i, Scalar(0, 255, 255), 2, 8);  //绘制
	}

	for (int i = 0; i < contours_poly.size(); i++)
	{
		std::vector<cv::Point>points_poly = contours_poly[i];
		for (int j = 0; j < points_poly.size(); j++)
		{			
			circle(contourImg, points_poly[j], 3, cv::Scalar(0, 0, 255));
			imshow("contourImg", contourImg);
			cv::waitKey(0);
		}	
	}
		
	std::vector<double>lineLength;
	for (int i = 0; i < lines.size(); i++)
	{				
		double line1Length = std::sqrt(std::pow(lines[i][0] - lines[i][2], 2) + std::pow(lines[i][1] - lines[i][3], 2));
		lineLength.push_back(line1Length);		
	}
	std::cout << lineLength.size() << "\n";
	for (int i = 0; i < lineLength.size(); i++)
	{
		std::cout << lineLength[i] << "   ";
	}
	std::cout << "\n";
	
	std::sort(lineLength.begin(), lineLength.end());
	std::vector<cv::Vec4i>lineContainer;
	for (int i = 0; i < lineLength.size(); i++)
	{
		std::cout << lineLength[i] << "   ";
		/*if (i == 4)
			break;*/
		for (int j = 0; j < lines.size(); j++)
		{
			if (std::sqrt(std::pow(lines[j][0] - lines[j][2], 2) + std::pow(lines[j][1] - lines[j][3], 2)) == lineLength[i])
			{
				std::cout << "与之对应的点：" << lines[i][0] << "  " << lines[i][1] << "  " << lines[i][2] << "  " << lines[i][3] << "\n";
				
				lineContainer.push_back(lines[i]);
				line(img, cv::Point2f(lines[i][0], lines[i][1]), cv::Point2f(lines[i][2], lines[i][3]),
					cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 2);
				imshow("img", img);
				cv::waitKey(0);
			}
		}
	}	

	std::cout << "\n";	
	cv::waitKey(0);
}

void LinesDetect(cv::Mat image/*,std::string str*/)
{
	DoEdgeDetect(image/*, str*/);
}

//将rgb转化为Lab
cv::Mat RGBToLab(cv::Mat src)
{
	cv::Mat_<cv::Vec3f>I = src;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			double L = 0.3811*I(i, j)[0] + 0.5783*I(i, j)[1] + 0.0402*I(i, j)[2];
			double M = 0.1967*I(i, j)[0] + 0.7244*I(i, j)[1] + 0.0782*I(i, j)[2];
			double S = 0.0241*I(i, j)[0] + 0.1288*I(i, j)[1] + 0.8444*I(i, j)[2];

			if (L == 0) L = 1;
			if (M == 0) M = 1;
			if (S == 0) S = 1;

			L = std::log(L)/std::log(10);
			M = std::log(M)/std::log(10);
			S = std::log(S)/std::log(10);

			I(i, j)[0] = (L + M + S) / std::sqrt(3.0);
			I(i, j)[1] = (L + M - 2 * S) / std::sqrt(6.0);
			I(i, j)[2] = (L - M) / std::sqrt(2.0);

		}
	}
	return I;
}

std::vector<double> computeMeans(cv::Mat m)
{
	double sum[3] = { 0 };
	int pixes = m.cols * m.rows;
	std::vector<double> means;
	means.resize(3);
	cv::Mat_<cv::Vec3f> I = m;
	
	for (int i = 0; i < I.rows; ++i)
	{
		for (int j = 0; j < I.cols; ++j)
		{
			for (int k = 0; k < 3; k++)
			{
				sum[k] += I(i, j)[k];
			}
		}
	}

	for (int i = 0; i < 3; i++)
	{
		means[i] = sum[i] / pixes;
	}

	return means;
}

std::vector<double>computeVariances(cv::Mat m, std::vector<double>means)
{
	double sum[3] = { 0 };
	int pixes = m.cols * m.rows;
	cv::Mat_<cv::Vec3f> I = m;
	std::vector<double> variances;
	variances.resize(3);
	
	for (int i = 0; i < I.rows; ++i)
	{
		for (int j = 0; j < I.cols; ++j)
		{
			for (int chanel = 0; chanel < 3; chanel++)
			{
				sum[chanel] += abs(I(i, j)[chanel] - means[chanel]);
			}
		}
	}

	for (int i = 0; i < 3; i++)
	{
		variances[i] = sqrt(sum[i] / pixes);
	}

	return variances;
}

cv::Mat LabToRGB(cv::Mat m)
{
	cv::Mat_<cv::Vec3f> I = m;
	for (int i = 0; i < I.rows; ++i)
	{
		for (int j = 0; j < I.cols; ++j)
		{
			double L = I(i, j)[0] / sqrt(3.0) + I(i, j)[1] / sqrt(6.0) + I(i, j)[2] / sqrt(2.0);
			double M = I(i, j)[0] / sqrt(3.0) + I(i, j)[1] / sqrt(6.0) - I(i, j)[2] / sqrt(2.0);
			double S = I(i, j)[0] / sqrt(3.0) - 2 * I(i, j)[1] / sqrt(6.0);

			L = std::pow(10,L);
			M = std::pow(10,M);
			S = std::pow(10,S);

			I(i, j)[0] = 4.4679*L - 3.5873*M + 0.1193*S;
			I(i, j)[1] = -1.2186*L + 2.3809*M - 0.1624*S;
			I(i, j)[2] = 0.0497*L - 0.2439*M + 1.2045*S;

			if (I(i, j)[0] > 255) I(i, j)[0] = 255;
			if (I(i, j)[0] < 0)	I(i, j)[0] = 0;

			if (I(i, j)[1] > 255) I(i, j)[1] = 255;
			if (I(i, j)[1] < 0)	I(i, j)[1] = 0;

			if (I(i, j)[2] > 255) I(i, j)[2] = 255;
			if (I(i, j)[2] < 0)	I(i, j)[2] = 0;
		}
	}
	return I;
}

void computeResult(cv::Mat resultImg, cv::Scalar srcVariances, cv::Scalar targetVariances, cv::Scalar srcMeans, cv::Scalar targetMeans,std::string writeName)
{
	cv::Mat_<cv::Vec3f> I = resultImg;
	double dataTemp[3] = { 0 };
	
	for (int chanel = 0; chanel < 3; chanel++)
	{
		dataTemp[chanel] = targetVariances[chanel] / srcVariances[chanel];
	}

	for (int i = 0; i < I.rows; ++i)
	{
		for (int j = 0; j < I.cols; ++j)
		{
			for (int chanel = 0; chanel < 3; chanel++)
			{
				I(i, j)[chanel] = dataTemp[chanel] * (I(i, j)[chanel] - srcMeans[chanel]) + targetMeans[chanel];
			}
		}
	}
	resultImg = LabToRGB(resultImg);
	resultImg.convertTo(resultImg, CV_8U, 255.0, 1 / 255.0);
	std::string filename = "F:\\迅雷下载\\imageEnhance\\"+writeName;
	imwrite(filename, resultImg);
}

void ColorTransfer(cv::Mat src, cv::Mat target,std::string writeName)
{
	cv::Mat srcImg_32F;
	cv::Mat resultImg;
	cv::Mat targetImg_32F;
	src.convertTo(srcImg_32F, CV_32FC3, 1.0f / 255.f);
	target.convertTo(targetImg_32F, CV_32FC3, 1.0f / 255.f);
	resultImg = srcImg_32F;

	cv::Mat srcImg_Lab = RGBToLab(srcImg_32F);
	cv::Mat targetImg_Lab = RGBToLab(targetImg_32F);

	/*for (int i = 0; i < targetImg_Lab.rows; i++)
	{
		for (int j = 0; j < targetImg_Lab.cols; j++)
		{
			std::cout << double(targetImg_Lab.at<Vec3f>(i, j)[0]) << "\n";
			std::cout << double(targetImg_Lab.at<Vec3f>(i, j)[1])<< "\n";
			std::cout << double(targetImg_Lab.at<Vec3f>(i, j)[2])<< "\n";
		}
	}*/

	cv::Scalar srcMean;
	cv::Scalar srcDev;
	cv::Scalar targetMean;
	cv::Scalar targetDev;
	meanStdDev(srcImg_Lab, srcMean, srcDev);
	meanStdDev(targetImg_Lab, targetMean, targetDev);

	computeResult(resultImg, srcDev, targetDev, srcMean, targetMean, writeName);
}

void detectImage(cv::Mat image, cv::Mat imageGray, cv::Mat imageCanny)
{
	std::vector<cv::Vec4i>lines;
	cv::Mat resizeImgae;
	cv::Mat midImage;
	cv::Mat resizeImage_1, resizeImage_2, resizeImage_3;
	HoughLinesP(imageCanny, lines, 1, CV_PI / 180, 80, 100, 100);
	std::cout << "Lines before: " << lines.size() << "\n";

	int cannyThreshold = 80;
	float factor = 2.5;
	const int maxLinesNum = 10;//最多检测出的直线数

	for (int i = 0; i < lines.size(); i++)
	{
		line(image, cv::Point2f(lines[i][0], lines[i][1]), cv::Point2f(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 255),5);
	}
	resize(image, resizeImage_3, cv::Size(image.rows / 5, image.cols / 5));
	imshow("image", resizeImage_3);

	while (lines.size() >= maxLinesNum)
	{
		cannyThreshold += 2;
		cv::Canny(imageGray, midImage, cannyThreshold, cannyThreshold*factor);
		cv::threshold(midImage, midImage, 128, 255, cv::THRESH_BINARY);
		cv::HoughLinesP(midImage, lines, 1, CV_PI / 180, 80, 100, 100);
	}

	std::cout << "cannyThreshold1: " << cannyThreshold << "\n";

	cv::Canny(imageGray, midImage, cannyThreshold, cannyThreshold*factor);
	cv::resize(image, resizeImage_1, cv::Size(image.rows / 5, image.cols / 5));
	cv::imshow("afterFilter", resizeImage_1);

	cv::threshold(midImage, midImage, 128, 255, cv::THRESH_BINARY);
	cv::resize(midImage, resizeImage_2, cv::Size(image.rows / 5, image.cols / 5));
	cv::imshow("afterThreshold", resizeImage_2);

	cv::HoughLinesP(midImage, lines, 1, CV_PI / 180, 80, 100, 100);
	std::cout << "Lines after: " << lines.size() << "\n";

	cv::waitKey(0);
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
	for (int i = 0; i < contours.size(); i++)
	{
		/*if (contourArea(contours[i]) == 0)
		continue;*/
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
	/*for (int i = area.size() - 1; i >= 0; i--)
	{
		double length = arcLength(contours[i], true);
		approxPolyDP(contours[i], outCurve, 0.02*length, true);

		if (length == contoursLength[contoursLength.size() - 1])
		{
			for (int j = 0; j < outCurve.size(); j++)
			{
				std::cout << outCurve[j].x << "   " << outCurve[j].y << "\n";
			}
		}
	}*/
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
	//std::string image= "F:\\迅雷下载\\SourceImg\\IMG_9374.JPG";
	//Mat sourceImg = imread(image);
	std::vector<std::string>ImageList;
	SearchImage(filename, ImageList);
	for (int i =2; i < ImageList.size(); i++)
	{
		int pos = 0;
		pos = ImageList[i].rfind("I");
		if (-1 == pos)
			continue;
		std::string name_= ImageList[i].substr(pos, 13);
		std::string imageName =ImageList[i]; /* "IMG_9485.JPG";*/

		//std::string imageName = "IMG_9376.jpg";
		cv::Mat dilateImage;
		cv::Mat erodeImage;
		//std::string fileName = "F:\\迅雷下载\\SourceImg\\";
		std::string name = /*filename + "\\" +*/ imageName;
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
		//resize(imageGray, midImage, Size(image.rows / 5, image.cols / 5));
		//cv::imshow("ImgResize", midImage);

		//GaussianBlur(image, image, Size(3, 3),0,0);
		//cv::Canny(imageGray, imageCanny, 30, 180);
		Canny(imageGray, imageCanny, 30, 80);
		//resize(imageCanny, img, Size(image.rows / 5, image.cols / 5));
		//cv::imshow("img", img);
		//cv::waitKey(0);

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
			//imshow("rotateImg", rotateImg);
			cv::imwrite("F:\\迅雷下载\\RedeceParems\\" + name_, rotateImg);
			out << "Method:面积最大的点围成的多边形: " <<  name_<<"\n";
			/*resize(circleImage, circleImage, Size(circleImage.rows / 5, circleImage.rows / 5));
			cv::imshow("circle", circleImage);
			cv::waitKey(0);*/
		}		
		else
		{		
			//std::vector<Point2f>points_ = detectPoints(curves, outCurve, image);
			cv::Mat dilateImage1;
			cv::Mat elements = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
			dilate(imageCanny, dilateImage, elements, cv::Point(-1, -1));
			//resize(dilateImage, dilateImage1, Size(circleImage.rows / 5, circleImage.rows / 5));
			//imshow("dilateImage", dilateImage1);
			//cv::waitKey(0);

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
				//imshow("rotateImg", rotateImg);
				cv::imwrite("F:\\迅雷下载\\RedeceParems\\" + name_, rotateImg);
				out << "Method:膨胀后面积最大的点围成的多边形: " << name_<<"\n";
				/*resize(circleImage, circleImage, Size(circleImage.rows / 5, circleImage.rows / 5));
				cv::imshow("circle", circleImage);
				cv::waitKey(0);*/
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
				cv::imwrite("F:\\迅雷下载\\RedeceParems\\" + name_, rotateImg);
				out << "Method:膨胀后周长最大的点围成的多边形: " << name_ << "\n";

			}
		}
	}
}

void method2()
{
	std::string filename = "F:\\迅雷下载\\SourceImg";
	//std::string image= "F:\\迅雷下载\\SourceImg\\IMG_9374.JPG";
	//Mat sourceImg = imread(image);
	std::vector<std::string>ImageList;
	SearchImage(filename, ImageList);
	for (int i =0; i < ImageList.size(); i++)
	{
		int pos = 0;
		pos = ImageList[i].rfind("I");
		if (-1 == pos)
			continue;
		std::string name_= ImageList[i].substr(pos, 13);
		std::string imageName =ImageList[i];  /*"IMG_9485.JPG"*/

		//std::string imageName = "IMG_9376.jpg";
		cv::Mat dilateImage;
		cv::Mat erodeImage;
		//std::string fileName = "F:\\迅雷下载\\SourceImg\\";
		//std::string name = filename + "\\" + imageName;
		cv::Mat imageGray;
		cv::Mat imageCanny;
		cv::Mat midImage;
		cv::Mat image = cv::imread(imageName);
		cv::Mat img;
		cv::Mat lineImage;
		cv::Mat lineFilterImage = image.clone();
		cv::cvtColor(image, imageGray, CV_BGR2GRAY);

		cv::Mat circleImage = image.clone();
		cv::medianBlur(imageGray, imageGray, 33);
		/*resize(imageGray, midImage, Size(image.rows / 5, image.cols / 5));
		cv::imshow("ImgResize", midImage);
*/
		//GaussianBlur(image, image, Size(3, 3),0,0);
		//cv::Canny(imageGray, imageCanny, 30, 180);
		cv::Canny(imageGray, imageCanny, 30, 80);
		/*resize(imageCanny, img, Size(image.rows / 5, image.cols / 5));
		cv::imshow("img", img);
		cv::waitKey(0);*/

		std::vector<std::vector<cv::Point>>contours;
		std::vector<cv::Point>outCurve;
		std::vector<std::vector<cv::Point>>curves;

		std::vector<double>contoursLength;
		std::vector<double>areaNum;

		{
			//std::vector<Point2f>points_ = detectPoints(curves, outCurve, image);
			cv::Mat dilateImage1;
			cv::Mat elements = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
			cv::dilate(imageCanny, dilateImage, elements, cv::Point(-1, -1));
		/*	resize(dilateImage, dilateImage1, Size(circleImage.rows / 5, circleImage.rows / 5));
			imshow("dilateImage", dilateImage1);
			cv::waitKey(0);*/

			detectFindContours(dilateImage, outCurve, curves, contoursLength, areaNum, contours, true);
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
				//imshow("rotateImg", rotateImg);
				cv::imwrite("F:\\迅雷下载\\TargetImage\\" + name_, rotateImg);
				/*resize(circleImage, circleImage, Size(circleImage.rows / 5, circleImage.rows / 5));
				cv::imshow("circle", circleImage);
				cv::waitKey(0);*/
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
				cv::imwrite("F:\\迅雷下载\\TargetImage\\" + name_, rotateImg);

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
	//imshow("sauvola", tempImg);
	std::string writeName = "F:\\迅雷下载\\imageEnhance\\"+ imgName;
	std::string writeName2 = "F:\\迅雷下载\\imageEnhance\\" + grayImgName;
	cv::imwrite(writeName2, tempImg);
	cv::imwrite(writeName, img);
	//waitKey(0);
}

void ImageEnhance()
{
	std::string filename = "F:\\迅雷下载\\RedeceParems";
	std::string imageName = "F:\\迅雷下载\\599322574318121290.jpg";
	cv::Mat img2_clahe;
	cv::Mat adaptiveThreshold;
	cv::Mat img1 = cv::imread(imageName);
	std::vector<std::string>ImageList;
	SearchImage(filename, ImageList);
	for (int i = 31; i < ImageList.size(); i++)
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
		double x= kernel_width/2 ;
		//std::cout << x;

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
	//method1();
	//method2();
	ImageReProcess();
	//ImageEnhance();//图像增强

	/*Mat img1(Size(2200, 2800), CV_8UC3, Scalar(200, 200, 200));
	imwrite("F:\\迅雷下载\\temp.jpg", img1);*/

	return 0;
}