#include "stdafx.h"

#include "GLDisp.h"


void DisplayMCML(){
	DisplayArray();
	DisplayPhoton();
}
void DisplayArray(){
	glBegin(GL_LINES);
	glColor3d(0.0, 0.0, 0.0);		//�F�̐ݒ�
	glEnd();
}
void DisplayPhoton(){
	glPushMatrix();
	glColor3d(1.0, 0.0, 0.0);		//�F�̐ݒ�
	glTranslated(0.0, 10.0, 20.0);	//���s�ړ��l�̐ݒ�
	glutSolidSphere(1.0, 10, 10);	//�����F(���a, Z���܂��̕�����, Z���ɉ�����������)
	glPopMatrix();
}
