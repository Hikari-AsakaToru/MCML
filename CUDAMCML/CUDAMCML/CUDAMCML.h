
// CUDAMCML.h : PROJECT_NAME アプリケーションのメイン ヘッダー ファイルです。
//

#pragma once

#ifndef __AFXWIN_H__
	#error "PCH に対してこのファイルをインクルードする前に 'stdafx.h' をインクルードしてください"
#endif

#include "resource.h"		// メイン シンボル


// CCUDAMCMLApp:
// このクラスの実装については、CUDAMCML.cpp を参照してください。
//

class CCUDAMCMLApp : public CWinApp
{
public:
	CCUDAMCMLApp();

// オーバーライド
public:
	virtual BOOL InitInstance();

// 実装

	DECLARE_MESSAGE_MAP()
};

extern CCUDAMCMLApp theApp;