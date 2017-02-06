#pragma once 
#include "stdafx.h"
class BaseGrap{
protected:
	void BlackCylinder(const float radius, const float height, const int sides, const float4* Color){
		// https://www21.atwiki.jp/opengl/pages/69.html 引用
		glColor4d(Color->x, Color->y, Color->z, Color->w); //色の設定
		double pi = 3.1415;
		//上面
		glNormal3d(0.0, 1.0, 0.0);
		glBegin(GL_POLYGON);
		for (double i = 0; i < sides; i++) {
			double t = pi * 2 / sides * (double)i;
			glVertex3d(radius * cos(t), height, radius * sin(t));
		}
		glEnd();
		// 上面
		glColor4d(0.0, 0.0, 0.0, 1.0); //色の設定
		glBegin(GL_LINE_STRIP);
		for (double i = 0; i < sides; i++) {
			double t = pi * 2 / sides * (double)i;
			glVertex3d(radius * cos(t), height, radius * sin(t));
		}
		glEnd();
		//側面
		glColor4d(Color->x, Color->y, Color->z, Color->w); //色の設定
		glBegin(GL_QUAD_STRIP);
		for (double i = 0; i <= sides; i = i + 1){
			double t = i * 2 * pi / sides;
			glNormal3f((GLfloat)cos(t), 0.0, (GLfloat)sin(t));
			glVertex3f((GLfloat)(radius*cos(t)), -height, (GLfloat)(radius*sin(t)));
			glVertex3f((GLfloat)(radius*cos(t)), height, (GLfloat)(radius*sin(t)));
		}
		glEnd();
		//下面
		glNormal3d(0.0, -1.0, 0.0);
		glColor4d(Color->x, Color->y, Color->z, Color->w); //色の設定
		glBegin(GL_POLYGON);
		for (double i = sides; i >= 0; --i) {
			double t = pi * 2 / sides * (double)i;
			glVertex3d(radius * cos(t), -height, radius * sin(t));
		}
		glEnd();
		glColor4d(0.0, 0.0, 0.0, 1.0); //色の設定
		glBegin(GL_LINE_STRIP);
		for (double i = sides; i >= 0; --i) {
			double t = pi * 2 / sides * (double)i;
			glVertex3d(radius * cos(t), -height, radius * sin(t));
		}
		glEnd();
	}
	void Cylinder(const float radius, const float height, const int sides){
		// https://www21.atwiki.jp/opengl/pages/69.html 引用
		double pi = 3.1415;
		//上面
		glNormal3d(0.0, 1.0, 0.0);
		glBegin(GL_POLYGON);
		for (double i = 0; i < sides; i++) {
			double t = pi * 2 / sides * (double)i;
			glVertex3d(radius * cos(t), height, radius * sin(t));
		}
		glEnd();
		//側面
		glBegin(GL_QUAD_STRIP);
		for (double i = 0; i <= sides; i = i + 1){
			double t = i * 2 * pi / sides;
			glNormal3f((GLfloat)cos(t), 0.0, (GLfloat)sin(t));
			glVertex3f((GLfloat)(radius*cos(t)), -height, (GLfloat)(radius*sin(t)));
			glVertex3f((GLfloat)(radius*cos(t)), height, (GLfloat)(radius*sin(t)));
		}
		glEnd();
		//下面
		glNormal3d(0.0, -1.0, 0.0);
		glBegin(GL_POLYGON);
		for (double i = sides; i >= 0; --i) {
			double t = pi * 2 / sides * (double)i;
			glVertex3d(radius * cos(t), -height, radius * sin(t));
		}
		glEnd();
	}

};

class MCMLGrap : public BaseGrap{
	MemStruct* DrawData;
	
public:
	inline MCMLGrap(){

	};
	~MCMLGrap(){};
	inline void SetData(MemStruct* hostMem){
		if (hostMem == NULL){
			return;
		}
		DrawData = hostMem;
	}
	inline void DispXYZMainRay(){
		glLineWidth(3.0);
		glBegin(GL_LINES);
		// X軸
		glColor3f(1.0, 0.0, 0.0);
		glVertex3f(-1.0, 0.0, 0.0);
		glVertex3f( 1.0, 0.0, 0.0);
		// Y軸
		glColor3f(0.0, 1.0, 0.0);
		glVertex3f(0.0, -1.0, 0.0);
		glVertex3f( 0.0, 1.0, 0.0);
		// Z軸
		glColor3f(0.0, 0.0, 1.0);
		glVertex3f(0.0, 0.0, -1.0);
		glVertex3f( 0.0, 0.0, 1.0);

		glEnd();
	};
	inline void DispXYZSubRay(const CString Ray){
		glLineWidth(0.01);
		glBegin(GL_LINES);
		for (int i = 0; i < 11; i++){
			float t= i*0.2 - 1.0;
			if (t*t > 0.0001){
				if (Ray == CString("XY")){
					glColor3f(0.5, 0.5, 0.0);
					glVertex3f(-1.0, t, 0.0);
					glVertex3f(1.0, t, 0.0);
				}
				if (Ray == CString("XZ")){
					glColor3f(0.5, 0.0, 0.5);
					glVertex3f(-1.0, 0.0,t);
					glVertex3f(1.0, 0.0 ,t);
				}
				if (Ray == CString("YX")){
					glColor3f(0.5, 0.5, 0.0);
					glVertex3f(t, -1.0, 0.0);
					glVertex3f(t, 1.0, 0.0);
				}
				if (Ray == CString("YZ")){
					glColor3f(0.0, 0.5, 0.5);
					glVertex3f(0.0, -1.0,t);
					glVertex3f(0.0, 1.0, t);
				}
				if (Ray == CString("ZX")){
					glColor3f(0.5, 0.0, 0.50);
					glVertex3f(t, 0.0, -1.0);
					glVertex3f(t, 0.0, 1.0);
				}
				if (Ray == CString("ZY")){
					glColor3f(0.0, 0.5, 0.50);
					glVertex3f(0.0,t, -1.0);
					glVertex3f(0.0,t, 1.0);
				}
			}
		}
		glEnd();
	}
	inline void DispXYZSubRayALL(){
		DispXYZSubRay(CString("XY"));
		DispXYZSubRay(CString("XZ"));
		DispXYZSubRay(CString("YX"));
		DispXYZSubRay(CString("YZ"));
		DispXYZSubRay(CString("ZX"));
		DispXYZSubRay(CString("ZY"));
	}
	inline void DispPhoton(){
		if (DrawData == NULL){
			return;
		}
		PhotonStruct* Input = DrawData->p;
		glPushMatrix();
		glColor3d(1.0, 0.5, 0.0); //色の設定
		GLUquadric* quadric = gluNewQuadric();
		glTranslatef((*Input).x, (*Input).y, (*Input).z);
		gluSphere(quadric, 0.025, 360, 360);
		gluDeleteQuadric(quadric);
		glPopMatrix();

	}
	inline void DispPhotonVec(){
		if (DrawData == NULL){
			return;
		}
		PhotonStruct* Input = DrawData->p;
		glPushMatrix();
		glColor3d(1.0, 1.0, 1.0); //色の設定
		glBegin(GL_LINE);
		glVertex3f((*Input).x, (*Input).y, (*Input).z);
		glVertex3f((*Input).x + (*Input).dx, (*Input).y + (*Input).dy,(*Input).z+(*Input).dz);
		glEnd();
		glPopMatrix();

	}
	inline void DispRecv(){
		if (DrawData == NULL){
			return;
		}

		glPushMatrix();
		float4 RGBA;
		RGBA.x = 1.0;
		RGBA.y = 1.0;
		RGBA.z = 0.0;
		RGBA.w = 0.1;

		BlackCylinder(0.5, 0.5, 360,&RGBA);
		glPopMatrix();

	}
};

