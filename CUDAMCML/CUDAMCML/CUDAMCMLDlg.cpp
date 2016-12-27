
// CUDAMCMLDlg.cpp : 実装ファイル
//

#define _VisualCPP_

#include "stdafx.h"
#include "CUDAMCML.h"
#include "CUDAMCMLDlg.h"
#include "afxdialogex.h"
#include <string>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CCUDAMCMLDlg ダイアログ



CCUDAMCMLDlg::CCUDAMCMLDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CCUDAMCMLDlg::IDD, pParent)
	, m_xvPath(_T("C:\\Users\\MacOS\\Documents\\Visual Studio 2013\\Projects\\InputData\\sample.mci"))
	, m_xvStatus(_T(""))
	, m_xvRefBase32(_T("C:\\Users\\MacOS\\Documents\\Visual Studio 2013\\Projects\\InputData\\safeprimes_base32.txt"))
	, m_xvSeed(100)
	, m_xvSeedText(_T("100"))
	, m_xvRadioFocus(0)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CCUDAMCMLDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT1, m_xvPath);
	DDV_MaxChars(pDX, m_xvPath, 100);
	DDX_Text(pDX, IDC_PROCESS_STATE, m_xvStatus);
	DDV_MaxChars(pDX, m_xvStatus, 100);
	DDX_Text(pDX, IDC_EDIT2, m_xvRefBase32);
	DDV_MaxChars(pDX, m_xvRefBase32, 200);
	DDX_Text(pDX, IDC_SEED, m_xvSeedText);
	DDX_Radio(pDX, IDC_RADIO1, m_xvRadioFocus);
}

BEGIN_MESSAGE_MAP(CCUDAMCMLDlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDOK, &CCUDAMCMLDlg::OnBnClickedOk)
	ON_BN_CLICKED(IDC_Ref, &CCUDAMCMLDlg::OnBnClickedRef)

	ON_BN_CLICKED(IDC_REF_BASE32, &CCUDAMCMLDlg::OnBnClickedRefBase32)
	ON_EN_CHANGE(IDC_EDIT2, &CCUDAMCMLDlg::OnEnChangeEdit2)
	ON_BN_CLICKED(IDC_RADIO1, &CCUDAMCMLDlg::OnBnClickedRadio1)
	ON_BN_CLICKED(IDC_RADIO2, &CCUDAMCMLDlg::OnBnClickedRadio2)
	ON_EN_CHANGE(IDC_PROCESS_STATE2, &CCUDAMCMLDlg::OnEnChangeProcessState2)

	ON_EN_KILLFOCUS(IDC_SEED, &CCUDAMCMLDlg::OnkillSeed)
END_MESSAGE_MAP()


// CCUDAMCMLDlg メッセージ ハンドラー

BOOL CCUDAMCMLDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// このダイアログのアイコンを設定します。アプリケーションのメイン ウィンドウがダイアログでない場合、
	//  Framework は、この設定を自動的に行います。
	SetIcon(m_hIcon, TRUE);			// 大きいアイコンの設定
	SetIcon(m_hIcon, FALSE);		// 小さいアイコンの設定

	// TODO: 初期化をここに追加します。
	m_uin64ErrorFlag = _No_ERROR_;
	return TRUE;  // フォーカスをコントロールに設定した場合を除き、TRUE を返します。
}

// ダイアログに最小化ボタンを追加する場合、アイコンを描画するための
//  下のコードが必要です。ドキュメント/ビュー モデルを使う MFC アプリケーションの場合、
//  これは、Framework によって自動的に設定されます。

void CCUDAMCMLDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 描画のデバイス コンテキスト

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// クライアントの四角形領域内の中央
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// アイコンの描画
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// ユーザーが最小化したウィンドウをドラッグしているときに表示するカーソルを取得するために、
//  システムがこの関数を呼び出します。
HCURSOR CCUDAMCMLDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CCUDAMCMLDlg::OnBnClickedOk()
{
	m_xvStatus =	m_cCUDAMCML.StartSim(&m_xvPath, m_xvPath.GetLength(), &m_xvRefBase32, m_xvRefBase32.GetLength(),m_xvSeed);
	UpdateData(FALSE);
	// TODO: ここにコントロール通知ハンドラー コードを追加します。
	// CDialogEx::OnOK();
}

void CCUDAMCMLDlg::OnBnClickedRef()
{
	// TODO: ここにコントロール通知ハンドラー コードを追加します。
	CString filter("INPUT FILE?|*.mci||");
	CFileDialog cSelectDlg(TRUE,0,0,OFN_HIDEREADONLY,filter);
	
	if (cSelectDlg.DoModal() == IDOK){
		m_csFileNAME = cSelectDlg.GetPathName();
		m_xvPath = cSelectDlg.GetPathName();
		UpdateData(FALSE);

	}

	return;
}
// MCML のIO部分

int cMCML::WriteSimRslts(MemStruct* HostMem, SimulationStruct* sim)
{
	CString TransCstr(sim->outp_filename);
	CStdioFile  cOutputFile(TransCstr, CFile::modeWrite | CFile::shareDenyNone | CFile::modeCreate);

	// Copy stuff from sim->det to make things more readable:
	double dr = (double)sim->det.dr;		// Detection grid resolution, r-direction [cm]
	double dz = (double)sim->det.dz;		// Detection grid resolution, z-direction [cm]
	double da = PI / (2 * sim->det.na);		// Angular resolution [rad]?

	int na = sim->det.na;			// Number of grid elements in angular-direction [-]
	int nr = sim->det.nr;			// Number of grid elements in r-direction
	int nz = sim->det.nz;			// Number of grid elements in z-direction


	int rz_size = nr*nz;
	int ra_size = nr*na;
	int r, a, z;
	unsigned int l;
	int i;

	unsigned long long temp = 0;
	double scale1 = (double)0xFFFFFFFF * (double)sim->number_of_photons;
	double scale2;



	cOutputFile.WriteString(_T("##### Data categories include: \n"));
	cOutputFile.WriteString(_T("# InParm, RAT, \n"));
	cOutputFile.WriteString(_T("# Rd_ta\t####\n\n"));
	cOutputFile.WriteString(_T("InParm \t\t\t# Input parameters.cm is used.\n"));
	cOutputFile.WriteString(TransCstr + _T("\tA\t# output file name, ASCII.\n"));
	// Duration Time (Process time)
	TransCstr.Format(_T("%d"), sim->RecordDoSim.procTotalTime);
	cOutputFile.WriteString(TransCstr + _T(" ,\t\t\t# Process Total   Time[ms]				\n"));
	// Duration Time (Process time)
	TransCstr.Format(_T("%d"), sim->RecordDoSim.procAvergTime);
	cOutputFile.WriteString(TransCstr + _T(" ,\t\t\t# Process Average Time[ms] (DoOneSim())	\n"));
	// Num of Photon
	TransCstr.Format(_T("%d"), sim->number_of_photons);
	cOutputFile.WriteString(TransCstr + _T(" ,\t\t\t# No. of photons\n"));
	// dr [cm]
	TransCstr.Format(_T("%f"), sim->det.dr);
	cOutputFile.WriteString(TransCstr + _T(" ,\t\t# dr[cm]\n"));

	// dz [cm]
	TransCstr.Format(_T("%f"), sim->det.dz);
	cOutputFile.WriteString(TransCstr + _T(" ,\t\t# dz[cm]\n"));

	// na [-]
	TransCstr.Format(_T("%d"), sim->det.na);
	cOutputFile.WriteString(TransCstr + _T(" ,\t\t\t# na[-]\n"));

	// nr [-]
	TransCstr.Format(_T("%d"), sim->det.nr);
	cOutputFile.WriteString(TransCstr + _T(" ,\t\t\t# nr[-]\n"));

	// nz [-]
	TransCstr.Format(_T("%d"), sim->det.nz);
	cOutputFile.WriteString(TransCstr + _T(" ,\t\t\t# nz[-]\n"));

	// num of layer [-]
	TransCstr.Format(_T("%d"), sim->n_layers);
	cOutputFile.WriteString(TransCstr + _T(" ,\t\t\t# LayerNum \n"));

	cOutputFile.WriteString(_T("#Num,\tn,\tmua,\tmutr,\tg,\td,\n"));
	for (int nLoop =1; nLoop < sim->n_layers+1; nLoop++){
		// レイヤーパラメ
		TransCstr.Format(_T("%d"), nLoop);
		cOutputFile.WriteString(_T("Num.")+TransCstr + _T(",\t"));
		TransCstr.Format(_T("%3.2f"), sim->layers[nLoop].n);
		cOutputFile.WriteString(TransCstr + _T(",\t"));
		TransCstr.Format(_T("%3.2f"), sim->layers[nLoop].mua);
		cOutputFile.WriteString(TransCstr + _T(",\t"));
		TransCstr.Format(_T("%3.2f"), sim->layers[nLoop].mutr);
		cOutputFile.WriteString(TransCstr + _T(",\t"));
		TransCstr.Format(_T("%3.2f"), sim->layers[nLoop].g);
		cOutputFile.WriteString(TransCstr + _T(",\t"));
		TransCstr.Format(_T("%3.2f"), sim->layers[nLoop].z_max - sim->layers[nLoop].z_min);
		cOutputFile.WriteString(TransCstr + _T(",\n"));
	}
	cOutputFile.WriteString(_T("\nRAT #Reflectance, absorption transmission\n"));
	unsigned long long Rs = 0;	// Specular reflectance [-]
	unsigned long long Rd = 0;	// Diffuse reflectance [-]
	unsigned long long A = 0;		// Absorbed fraction [-]
	unsigned long long T = 0;		// Transmittance [-]
	Rs = (unsigned long long)(0xFFFFFFFFu - sim->start_weight)*(unsigned long long)sim->number_of_photons;
	for (i = 0; i < rz_size; i++){
		A += HostMem->A_rz[i];
	}
	for (i = 0; i<ra_size; i++){ 
		T += HostMem->Tt_ra[i];
		Rd += HostMem->Rd_ra[i];
	}


	TransCstr.Format(_T("%G \t\t #Specular reflectance [-]\n"), (double)Rs / scale1);
	cOutputFile.WriteString(TransCstr);
	TransCstr.Format(_T("%G \t\t #Diffuse reflectance [-]\n"), (double)Rd / scale1);
	cOutputFile.WriteString(TransCstr);
	TransCstr.Format(_T("%G \t\t #Absorbed fraction  [-]\n"), (double)A / scale1);
	cOutputFile.WriteString(TransCstr);
	TransCstr.Format(_T("%G \t\t #Transmittance [-]\n"), (double)T / scale1);
	cOutputFile.WriteString(TransCstr);


	// Calculate and write A_l
	TransCstr.Format(_T("\nA_l #Absorption as a function of layer. [-]\n"));
	cOutputFile.WriteString(TransCstr);
	z = 0;
	for (l = 1; l <= sim->n_layers; l++)
	{
		temp = 0;
		while (((double)z + 0.5)*dz <= sim->layers[l].z_max)
		{
			for (r = 0; r<nr; r++) temp += HostMem->A_rz[z*nr + r];
			z++;
			if (z == nz)break;
		}
		TransCstr.Format(_T("%G\n"), (double)temp / scale1);
		cOutputFile.WriteString(TransCstr);
	}

	// Calculate and write A_z
	scale2 = scale1*dz;
	cOutputFile.WriteString(_T("\nA_z #A[0], [1],..A[nz-1]. [1/cm]\n"));
	for (z = 0; z<nz; z++)
	{
		temp = 0;
		for (r = 0; r < nr; r++){
			temp += HostMem->A_rz[z*nr + r];
		}
		TransCstr.Format(_T("%E\n"), (double)temp / scale2);
		cOutputFile.WriteString(TransCstr);
	}

	cOutputFile.WriteString(_T("\n"));
	cOutputFile.WriteString(_T("#\t\tResult of kakuhan Rd_ra[nr][na]\n"));
	cOutputFile.WriteString(_T("#\t\tRd_ra[ 0][0],Rd_ra[ 0][1],....,Rd_ra[ 0][nr-1]\n"));
	cOutputFile.WriteString(_T("#\t\tRd_ra[ 1][0],Rd_ra[ 1][1],....,Rd_ra[ 1][nr-1]\n"));
	cOutputFile.WriteString(_T("#\t\t....,....,....,....\n"));
	cOutputFile.WriteString(_T("#\t\tRd_ra[na][0],Rd_ra[na][1],....,Rd_ra[na][nr-1]\n\n"));
	cOutputFile.WriteString(_T(","));
	for (int nLoopNR = 1; nLoopNR <= sim->det.nr; nLoopNR++){
		TransCstr.Format(_T("R%d,"), nLoopNR);
		cOutputFile.WriteString(TransCstr);
	}
	cOutputFile.WriteString(_T("\n"));
	// Rd_ra 出力
	for (int nLoopNA = 0; nLoopNA < sim->det.na; nLoopNA++){
		TransCstr.Format(_T("A%d,"), nLoopNA);
		cOutputFile.WriteString(TransCstr);
		for (int nLoopNR = 0; nLoopNR < sim->det.nr; nLoopNR++){
//			TransCstr.Format(_T("%1.8lf"), (double)HostMem->Rd_ra[nLoopNR + nLoopNA*sim->det.nr] / (m_un64NumPhoton*m_simulations[0].start_weight));
			TransCstr.Format(_T("%21u"), HostMem->Rd_ra[nLoopNR + nLoopNA*sim->det.nr] );
			cOutputFile.WriteString(TransCstr + _T(","));
		}
		cOutputFile.WriteString(_T("\n"));

	}
	cOutputFile.WriteString(_T("\n"));
	cOutputFile.WriteString(_T("#\t\tResult of touka Tt_ra[nr][na]\n"));
	cOutputFile.WriteString(_T("#\t\tTt_ra[ 0][0],Tt_ra[ 0][1],....,Tt_ra[ 0][nr-1]\n"));
	cOutputFile.WriteString(_T("#\t\tTt_ra[ 1][0],Tt_ra[ 1][1],....,Tt_ra[ 1][nr-1]\n"));
	cOutputFile.WriteString(_T("#\t\t....,....,....,....\n"));
	cOutputFile.WriteString(_T("#\t\tTt_ra[na][0],Tt_ra[na][1],....,Tt_ra[na][nr-1]\n\n"));
	cOutputFile.WriteString(_T(","));
	for (int nLoopNR = 1; nLoopNR <= sim->det.nr; nLoopNR++){
		TransCstr.Format(_T("R%d,"), nLoopNR);
		cOutputFile.WriteString(TransCstr);
	}
	cOutputFile.WriteString(_T("\n"));
	// Tt-Ra 出力
	for (int nLoopNA = 0; nLoopNA < sim->det.na; nLoopNA++){
		TransCstr.Format(_T("A%d,"), nLoopNA);
		cOutputFile.WriteString(TransCstr);
		for (int nLoopNR = 0; nLoopNR < sim->det.nr; nLoopNR++){
//			TransCstr.Format(_T("%1.8lf"), (double)HostMem->Tt_ra[nLoopNR + nLoopNA*sim->det.nr] / (m_un64NumPhoton*m_simulations[0].start_weight));
			TransCstr.Format(_T("%21u"), HostMem->Tt_ra[nLoopNR + nLoopNA*sim->det.nr]);
			cOutputFile.WriteString(TransCstr + _T(","));
		}
		cOutputFile.WriteString(_T("\n"));
	}
	cOutputFile.WriteString(_T("\n"));
	cOutputFile.WriteString(_T("#\t\tResult of A_rz[nr][na]\n"));
	cOutputFile.WriteString(_T("#\t\tA_rz[ 0][0],A_rz[ 0][1],....,A_rz[ 0][nr-1]\n"));
	cOutputFile.WriteString(_T("#\t\tA_rz[ 1][0],A_rz[ 1][1],....,A_rz[ 1][nr-1]\n"));
	cOutputFile.WriteString(_T("#\t\t....,....,....,....\n"));
	cOutputFile.WriteString(_T("#\t\tA_rz[na][0],A_rz[na][1],....,A_rz[na][nr-1]\n\n"));
	cOutputFile.WriteString(_T("\n"));
	cOutputFile.WriteString(_T(","));
	for (int nLoopNR = 1; nLoopNR <= sim->det.nr; nLoopNR++){
		TransCstr.Format(_T("r%d"), nLoopNR);
		cOutputFile.WriteString(TransCstr + _T(","));
	}
	cOutputFile.WriteString(_T("\n"));
	// A-Rz 出力
	for (int nLoopNZ = 0; nLoopNZ < sim->det.nz; nLoopNZ++){
		TransCstr.Format(_T("Z%d,"), nLoopNZ);
		cOutputFile.WriteString(TransCstr);
		for (int nLoopNR = 0; nLoopNR < sim->det.nr; nLoopNR++){
			TransCstr.Format(_T("%u"), HostMem->A_rz[nLoopNR + nLoopNZ*sim->det.nr]);
//			TransCstr.Format(_T("%1.8lf"), (double)HostMem->A_rz[nLoopNR + nLoopNZ*sim->det.nr] / (m_un64NumPhoton*m_simulations[0].start_weight));
			cOutputFile.WriteString(TransCstr + _T(","));
		}
		cOutputFile.WriteString(_T("\n"));
	}
	// Rd_ra

	for (int it = 0; it < sim->nr; it++)
	{
		for (int ia = 0; ia < HostMem->sim->na; ia++)
		{
			TransCstr.Format(_T("%12.4E,"), HostMem->Out_Ptr->Rd_ra[it + sim->det.nr*ia]);
			cOutputFile.WriteString(TransCstr);
			if ((it*HostMem->sim->na + ia + 1) % 9 == 0) cOutputFile.WriteString(_T("\n"));
		}
	}
	cOutputFile.WriteString(_T("\nStartWeight \n"));
	
	

	WriteOPL(&cOutputFile);
	WriteRd_p(&cOutputFile);
	WriteInParm(&cOutputFile,*sim);
	WriteRd_ra(&cOutputFile);
	

	
	return 0;

}

CString cMCML::ReadSimData(CString* filename, SimulationStruct** simulations, int ignoreAdetection,unsigned int Seed)
{
	int i = 0;
	int ii = 0;
	unsigned long number_of_photons;
	unsigned int start_weight;
	int n_simulations = 0;
	bool bOpenFlag;
	CString tmpStrData;
	CString cstrStatus;
	char mystring[STR_LEN];
	char str[STR_LEN];
	char AorB;
	float dtot = 0;
	
	if (!CheckGPU()){
		cstrStatus = "GPU未発見";
		::OutputDebugString(cstrStatus);
		return cstrStatus;
	}
	SimulationStruct stmpSimData;
	CStdioFile  cInputFile(*filename, CFile::modeRead|CFile::shareDenyNone);
	FileIO		cInputFileIO;

	cInputFileIO.SetFile(&cInputFile);
	// First read the first data line (file version) and ignore
	cInputFileIO.ReadFile1Flt(&m_fVer);
	cInputFileIO.ReadFile1UInt(&m_nRunCount);
	*simulations = new SimulationStruct[m_nRunCount];
	if (*simulations == NULL){ 
		cstrStatus="HostMemの確保失敗(SimStruct)\n";
		::OutputDebugString(cstrStatus);
		return cstrStatus;
	}
	CStringA TmpPathName(filename->GetBuffer(0));
	std::string TmpStrInp = TmpPathName.GetBuffer(0);



	for (int nLoop = 0; nLoop < m_nRunCount; nLoop++){

		cInputFileIO.ReadFile1CStr	( &m_cstrOutputName);
		cInputFileIO.ReadFile1UInt64( &(*simulations)[nLoop].number_of_photons);
		cInputFileIO.ReadFile1Flt	( &(*simulations)[nLoop].det.dz);
		cInputFileIO.ReadFile1Flt	( &(*simulations)[nLoop].det.dr);
		cInputFileIO.ReadFile1UInt	( &(*simulations)[nLoop].det.nz);
		cInputFileIO.ReadFile1UInt	( &(*simulations)[nLoop].det.nr);
		cInputFileIO.ReadFile1UInt	( &(*simulations)[nLoop].det.na);
		cInputFileIO.ReadFile1Dbl	( &(*simulations)[nLoop].r);		// SD
		cInputFileIO.ReadFile1Sht	( &(*simulations)[nLoop].nr);
		cInputFileIO.ReadFile1Sht	( &(*simulations)[nLoop].na);
		(*simulations)[nLoop].dt = 360 / (*simulations)[nLoop].nr;
		(*simulations)[nLoop].da = 90 / (*simulations)[nLoop].na;

		cInputFileIO.ReadFile1UInt	( &(*simulations)[nLoop].n_layers);

		(*simulations)[nLoop].Seed = Seed;
		// 入力ファイル名
		strcpy_s((*simulations)[nLoop].inp_filename, TmpStrInp.c_str());
		CStringA TmpOutPutName(m_cstrOutputName.GetBuffer(0));
		std::string TmpStrOtp = TmpOutPutName.GetBuffer(0);
		// 出力ファイル名
		strcpy_s((*simulations)[nLoop].outp_filename, TmpStrOtp.c_str());

		// モード選択
		(*simulations)[nLoop].ignoreAdetection = ignoreAdetection;

		// ファイル名の文字形式(ASCII)
		strcpy_s((*simulations)[nLoop].outp_filename, TmpStrOtp.c_str());
		(*simulations)[nLoop].AorB = (char)"A";


		//　一度に計算するフォトン数
//		(*simulations)[nLoop].number_of_photons = NUM_PHOTON_GPU;




		// レイヤー情報の為のメモリ確保
		(*simulations)[nLoop].layers = new LayerStruct[sizeof(LayerStruct)*((*simulations)[nLoop].n_layers + 2)];
		if ((*simulations)[nLoop].layers == NULL){ 
			cstrStatus = "HostMemの確保失敗(Layer)\n";
			::OutputDebugString(cstrStatus);
			return cstrStatus;
		}

		// 空気層の保存
		unsigned int tmpuiMed = 0;
		cInputFileIO.ReadFile1UInt( &tmpuiMed);
		(*simulations)[nLoop].layers[0].n = 1.0;
		(*simulations)[nLoop].layers[0].g = 0.9;
		(*simulations)[nLoop].layers[0].z_min = -9999.9;
		(*simulations)[nLoop].layers[0].z_max = 0.00;

		// レイヤー情報の保存
		dtot = 0;
		float tmpfD = 0;
		int tmpiMUS = 0;
		for (int LayerLoop = 1; LayerLoop <= (*simulations)[nLoop].n_layers; LayerLoop++)
		{
			cInputFileIO.ReadFile1Flt(&(*simulations)[nLoop].layers[LayerLoop].n);
			cInputFileIO.ReadFile1Flt(&(*simulations)[nLoop].layers[LayerLoop].mua);
			cInputFileIO.ReadFile1Int(&tmpiMUS);
			cInputFileIO.ReadFile1Flt(&(*simulations)[nLoop].layers[LayerLoop].g);
			cInputFileIO.ReadFile1Flt(&tmpfD);

			(*simulations)[nLoop].layers[LayerLoop].z_min	= dtot;
			dtot += tmpfD;
			(*simulations)[nLoop].layers[LayerLoop].z_max	= dtot;
			if (tmpiMUS == 0.0f){
				(*simulations)[nLoop].layers[LayerLoop].mutr = FLT_MAX; //Glas layer
			}
			else{
				(*simulations)[nLoop].layers[LayerLoop].mutr = 1.0f / ((*simulations)[nLoop].layers[LayerLoop].mua + tmpiMUS);
			}
		}
		// Refレイヤーの保存
		unsigned int tmpuiRef = 0;
		cInputFileIO.ReadFile1UInt(&tmpuiRef);
		(*simulations)[nLoop].layers[(*simulations)[nLoop].n_layers + 1].n = (float)tmpuiRef;

		(*simulations)[nLoop].end = 100;
		
		
		// スレッド数の決定
		// フォトンの数とメッシュ数に応じたGPUの必要メモリを算出
		UINT64 Tmp = (*simulations)[nLoop].number_of_photons;
		(*simulations)[nLoop].nDivedSimNum = 1;
		m_un64Membyte = (sizeof(PhotonStruct) + sizeof(unsigned long long) + 2 * sizeof(int))*(*simulations)[nLoop].number_of_photons + sizeof(unsigned int)
		+ sizeof(unsigned long long)*((*simulations)[nLoop].det.nr*((*simulations)[nLoop].det.nz + 2 * (*simulations)[nLoop].det.na));
		// 全体をGPU上に乗せることができるか？　 
		while ((m_un64Membyte > (unsigned long long)(m_sDevProp.totalGlobalMem*0.9)) && (((*simulations)[nLoop].number_of_photons / NUM_THREADS_PER_BLOCK + 1)>(m_sDevProp.multiProcessorCount-1))){
			// 不可能な場合=>フォトン数を半減=実行スレッド数を半減
			(*simulations)[nLoop].number_of_photons /= 2;
			// 分割数 
			(*simulations)[nLoop].nDivedSimNum *= 2;
			m_un64Membyte = (sizeof(PhotonStruct) + sizeof(unsigned long long) + 3 * sizeof(int))*(*simulations)[nLoop].number_of_photons
				+ sizeof(unsigned long long)*((*simulations)[nLoop].det.nr*((*simulations)[nLoop].det.nz + 2 * (*simulations)[nLoop].det.na));
		}
//		(*simulations)[nLoop].nDivedSimNum *= Tmp / (*simulations)[nLoop].number_of_photons;
		(*simulations)[nLoop].nDivedSimNum++;
		//　初期質量の計算
		double n1 = (*simulations)[nLoop].layers[0].n;
		double n2 = (*simulations)[nLoop].layers[1].n;
		double r = (n1 - n2) / (n1 + n2);
		r = r*r;
		start_weight = (unsigned long long)((double)0xffffffff * (1 - r));
		(*simulations)[nLoop].start_weight = start_weight;

	}

	return  cstrStatus=_T("NoProblem: Read File");
}

int cMCML::InitRNG(	const unsigned long long n_rng, CString *safeprimes_file, unsigned long long xinit)
{
	CStdioFile  cInputFile(*safeprimes_file, CFile::modeRead );
	unsigned int begin = 0u;
	unsigned int fora, tmp1, tmp2;
	unsigned long long * x = m_sHostMem.x;
	unsigned int * a = m_sHostMem.a;
	//if (strlen(safeprimes_file) == 0)
	//{
	//	// Try to find it in the local directory
	//	safeprimes_file = "safeprimes_base32.txt";
	//}


	// if (fp == NULL)
	// {
	// 	CString cstrStatus = _T("開けなかっす(safeprimes_base32.txt");
	// 	::OutputDebugString(cstrStatus);
	// 	return FALSE;
	// }
	CString tmpcstr;
	cInputFile.ReadString(tmpcstr);
	begin = _tcstoul(tmpcstr.Mid(0, 10), NULL,10);
	tmp1 = _tcstoul(tmpcstr.Mid(11, 31), NULL,10);
	tmp2 = _tcstoul(tmpcstr.Mid(31, 51), NULL,10);

	// fscanf(fp, "%u %u %u", &begin, &tmp1, &tmp2);

	// Here we set up a loop, using the first multiplier in the file to generate x's and c's
	// There are some restictions to these two numbers:
	// 0<=c<a and 0<=x<b, where a is the multiplier and b is the base (2^32)
	// also [x,c]=[0,0] and [b-1,a-1] are not allowed.

	//Make sure xinit is a valid seed (using the above mentioned restrictions)
	if ((xinit == 0ull) | (((unsigned int)(xinit >> 32)) >= (begin - 1)) | (((unsigned int)xinit) >= 0xfffffffful))
	{
		//xinit (probably) not a valid seed! (we have excluded a few unlikely exceptions)
		return 1;
	}

	for (unsigned int i = 0; i < n_rng; i++)
	{
		cInputFile.ReadString(tmpcstr);
		fora = _tcstoul(tmpcstr.Mid(0, 10), NULL,10);
		tmp1 = _tcstoul(tmpcstr.Mid(11, 31), NULL,10);
		tmp2 = _tcstoul(tmpcstr.Mid(32, 52), NULL,10);

		// fscanf(fp, "%u %u %u", &fora, &tmp1, &tmp2);
		a[i] = fora;
		x[i] = 0;
		while ((x[i] == 0) | (((unsigned int)(x[i] >> 32)) >= (fora - 1)) | (((unsigned int)x[i]) >= 0xfffffffful))
		{
			//generate a random number
			xinit = (xinit & 0xffffffffull)*(begin)+(xinit >> 32);

			//calculate c and store in the upper 32 bits of x[i]
			x[i] = (unsigned int)floor((((double)((unsigned int)xinit)) / (double)0x100000000)*fora);//Make sure 0<=c<a
			x[i] = x[i] << 32;

			//generate a random number and store in the lower 32 bits of x[i] (as the initial x of the generator)
			xinit = (xinit & 0xffffffffull)*(begin)+(xinit >> 32);//x will be 0<=x<b, where b is the base 2^32
			x[i] += (unsigned int)xinit;
		}
		//if(i<10)printf("%llu\n",x[i]);
	}
	cInputFile.Close();

	return 0;
}

CString cMCML::StartSim(CString* chPathName, int nPathName, CString* cstrB32Name, int B32NameLeng,unsigned int Seed)
{
	int i;
	// unsigned long long seed = (unsigned long long) time(NULL);// Default, use time(NULL) as seed
	int ignoreAdetection = 0;

	// シミュレーションデータの読み込み
	CString Status =ReadSimData(chPathName, &m_simulations, 0,Seed);
	CString Tmp2;
	if (Status != _T("NoProblem: Read File"))		{
		FreeFailedSimStrct(m_simulations, m_nRunCount);
		return Status;
	}
	unsigned int MemState=0;
	for (int nRun = 0; nRun < m_nRunCount; nRun++){

		// Device(GPU) とHost(CPU) のメモリ確保
		MemState = InitMallocMem(&m_simulations[nRun]);
		if (MemState != 0){
			FreeFailedSimStrct(&m_simulations[nRun], m_nRunCount);
			Tmp2.Format(_T("%x"), MemState);
			return _T("SimMem Malloc Error:") + Tmp2;
		}

		// Sim[Run] の定数のコピー
		MemState = InitDCMem(&m_simulations[nRun]);
		if (MemState != 0){
			FreeFailedSimStrct(&m_simulations[nRun], m_nRunCount);
			Tmp2.Format(_T("%x"), MemState);
			return _T("GPUMem Constant Error:") + Tmp2;
		}



		// Sim[Run] の初期化
		MemState = InitContentsMem(&m_simulations[nRun]);
		if (MemState != 0){
			FreeFailedSimStrct(&m_simulations[nRun], m_nRunCount);
			Tmp2.Format(_T("%x"), MemState);
			return _T("InitSim Error:") + Tmp2;
		}
		unsigned int InitState = 0;
		// フォトンの分割数に基づいて繰り返し
		auto Start = std::chrono::system_clock::now();
		InitOutput();
		
		for (int nDivNum = 0; nDivNum < m_simulations[nRun].nDivedSimNum; nDivNum++){
		int	 RunStatus = MakeRandTableDev();
			 RunStatus = InitPhoton();		
			 
			// Run a simulation
			RunStatus = DoOneSimulation(&m_simulations[nRun]);
			
			
			// cCUDAMCML::RunOldCarnel();

			if (RunStatus != 0){
				FreeSimulationStruct(m_simulations, m_nRunCount);
				Tmp2.Format(_T("%d"), RunStatus);
				return _T("RunSim Error:") + Tmp2;
			}
		}
		auto End = std::chrono::system_clock::now();
		auto AveDur = (End - Start) / m_simulations[nRun].nDivedSimNum;
		m_simulations[nRun].RecordDoSim.procAvergTime = std::chrono::duration_cast<std::chrono::milliseconds>(AveDur).count();
		m_simulations[nRun].RecordDoSim.procTotalTime = std::chrono::duration_cast<std::chrono::milliseconds>(End - Start).count();
		CopyDeviceToHostMem(&m_sHostMem, &m_sDeviceMem, &m_simulations[nRun]);
		//	シミュレーション結果を出力
		WriteSimRslts(&m_sHostMem, &m_simulations[nRun]);
		const char* tmp="sample.csv";
		strcpy_s(m_simulations[nRun].outp_filename, tmp);
		WriteSimRslts(&m_sHostMem, &m_simulations[nRun]);
	}
	// シミュレーションの終了処理
	FreeSimulationStruct(m_simulations, m_nRunCount);

	MessageBeep(-1);
	MessageBox(NULL,_T("Simulation OK!"),_T("Sim End"),MB_ICONQUESTION);
	return _T("Success Sim");
}

CString cMCML::CreateSimRsltFile(){
	CStdioFile Cc;
	return _T("出力完了");
}

void CCUDAMCMLDlg::OnBnClickedRefBase32()
{
	// TODO: ここにコントロール通知ハンドラー コードを追加します。
	// TODO: ここにコントロール通知ハンドラー コードを追加します。
	CString filter("Base32 FILE?|*.txt||");
	CFileDialog cSelectDlg(TRUE, 0, 0, OFN_HIDEREADONLY, filter);

	if (cSelectDlg.DoModal() == IDOK){
		m_csBase32NAME = cSelectDlg.GetPathName();
		m_xvRefBase32  = cSelectDlg.GetPathName();
		UpdateData(FALSE);

	}

	return;
}


void CCUDAMCMLDlg::OnEnChangeEdit2()
{
	// TODO: これが RICHEDIT コントロールの場合、このコントロールが
	// この通知を送信するには、CDialogEx::OnInitDialog() 関数をオーバーライドし、
	// CRichEditCtrl().SetEventMask() を
	// OR 状態の ENM_CHANGE フラグをマスクに入れて呼び出す必要があります。

	// TODO: ここにコントロール通知ハンドラー コードを追加してください。
}


void CCUDAMCMLDlg::OnBnClickedRadio1()
{
	// TODO: ここにコントロール通知ハンドラー コードを追加します。
	// シード値を現在の時刻を利用する場合
	CWnd* lpWnd = GetDlgItem(IDC_SEED);
	lpWnd->EnableWindow(FALSE);				// シード値代入用エディットボックスの無効化
	m_xvSeed = time(NULL);
}


void CCUDAMCMLDlg::OnBnClickedRadio2()
{
	// TODO: ここにコントロール通知ハンドラー コードを追加します。
	// シード値をこちらから指定する場合
	CWnd* lpWnd = GetDlgItem(IDC_SEED);
	lpWnd->EnableWindow(TRUE);					// シード値代入用エディットボックスの有効化
	wchar_t Tmp;
	wchar_t * ErrorCheck;
	unsigned int Ret;
	UpdateData();
	_tcstoul(m_xvSeedText, &ErrorCheck, 10);
	if (ErrorCheck == m_xvSeedText){
		m_uin64ErrorFlag |= _Err_NO_NUMERIC_;
		m_xvSeed = 0;
		MessageBox(_T("数値のみを入力してください"));
		return;
	}
	if (ERANGE == errno){
		m_uin64ErrorFlag |= _Err_NO_NUMERIC_;
		m_xvSeed = 0;
		MessageBox(_T("変換ミス"));
		return;
	}
	m_xvSeed = _tcstoul(m_xvSeedText, NULL, 10);

}


void CCUDAMCMLDlg::OnEnChangeProcessState2()
{
	// TODO: これが RICHEDIT コントロールの場合、このコントロールが
	// この通知を送信するには、CDialogEx::OnInitDialog() 関数をオーバーライドし、
	// CRichEditCtrl().SetEventMask() を
	// OR 状態の ENM_CHANGE フラグをマスクに入れて呼び出す必要があります。

	// TODO: ここにコントロール通知ハンドラー コードを追加してください。
}

void CCUDAMCMLDlg::OnkillSeed()
{
	// TODO: これが RICHEDIT コントロールの場合、このコントロールが
	// この通知を送信するには、CDialogEx::OnInitDialog() 関数をオーバーライドし、
	// EM_SETEVENTMASK メッセージを、
	// OR 状態の ENM_UPDATE フラグを lParam マスクに入れて、このコントロールに送信する必要があります。

	// TODO: ここにコントロール通知ハンドラー コードを追加してください。
	
	wchar_t Tmp;
	wchar_t * ErrorCheck;
	unsigned int Ret;
	UpdateData();
	_tcstoul(m_xvSeedText, &ErrorCheck, 10);
	if (ErrorCheck == m_xvSeedText){
		m_uin64ErrorFlag |= _Err_NO_NUMERIC_;
		m_xvSeed = 0;
		MessageBox(_T("数値のみを入力してください"));
		return;
	}
	if (ERANGE == errno){
		m_uin64ErrorFlag |= _Err_NO_NUMERIC_;
		m_xvSeed = 0;
		MessageBox(_T("変換ミス"));
		return;
	}
	m_xvSeed = _tcstoul(m_xvSeedText, NULL, 10);

}
