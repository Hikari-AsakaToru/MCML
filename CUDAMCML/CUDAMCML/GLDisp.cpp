#include "stdafx.h"

#include "GLDisp.h"


void DisplayMCML(){
	DisplayArray();
	DisplayPhoton();
}
void DisplayArray(){
	glBegin(GL_LINES);
	glColor3d(0.0, 0.0, 0.0);		//色の設定
	glEnd();
}
void DisplayPhoton(){
	glPushMatrix();
	glColor3d(1.0, 0.0, 0.0);		//色の設定
	glTranslated(0.0, 10.0, 20.0);	//平行移動値の設定
	glutSolidSphere(1.0, 10, 10);	//引数：(半径, Z軸まわりの分割数, Z軸に沿った分割数)
	glPopMatrix();
}
