
// CUDAMCMLDlg.h : �w�b�_�[ �t�@�C��
//

#pragma once
#include "CUDAMCML_IO.h"
#include "afxwin.h"
#include "MCMLGL.h"
#define _No_ERROR_			0x0000000000000000
#define _Wrong_				0x0000000080000000
#define _Err_NO_NUMERIC_	0x8000000100000000
#define _Err_OVER_FLOW_		0x8000000200000000


// CCUDAMCMLDlg �_�C�A���O
class CCUDAMCMLDlg : public CDialogEx
{
protected:

// �R���X�g���N�V����
public:
	CCUDAMCMLDlg(CWnd* pParent = NULL);	// �W���R���X�g���N�^�[

// �_�C�A���O �f�[�^
	enum { IDD = IDD_CUDAMCML_DIALOG };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV �T�|�[�g
	cMCML m_cCUDAMCML;
	CString m_csFileNAME;
	CString m_csBase32NAME;
	// �p�X�\���p
	CString m_xvPath;
	CString cLoadState;
	CString m_xvStatus;
	// �V�[�h�l

	// �G���[�`�F�b�N�p
	UINT64 m_uin64ErrorFlag; // 0 = No prob 0x80000000 00000000  Error
	// ����
protected:
	HICON m_hIcon;

	// �������ꂽ�A���b�Z�[�W���蓖�Ċ֐�
	virtual BOOL OnInitDialog();
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	

	afx_msg void OnBnClickedOk();
	afx_msg void OnBnClickedRef();
	afx_msg void OnClickedRef();

public:
	afx_msg void OnEnChangeEdit1();

	CString m_xvRefBase32;
	afx_msg void OnBnClickedRefBase32();
	afx_msg void OnEnChangeEdit2();
	afx_msg void OnBnClickedRadio1();
	afx_msg void OnBnClickedRadio2();
	unsigned int m_xvSeed;
	afx_msg void OnEnChangeProcessState2();
	CString m_xvSeedText;
	int m_xvRadioFocus;
	afx_msg void OnkillSeed();
private:
	CStatic m_Pic;
	CDC* m_pDC;
	HGLRC m_GLRC;
	bool SetupPixelFormat(HDC hdc);
	MCMLGrap Graphics;
public:
	afx_msg void OnDestroy();
	afx_msg void OnSize(UINT nType, int cx, int cy);
};


