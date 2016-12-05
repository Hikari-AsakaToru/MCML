#pragma once
#include "cuda_runtime_api.h"
#include "MFC_IO.h"
#include "CUDAMCML_GPGPU.h"





// C++で書かなきゃ無理な部分(Input/Output)は継承先で書く
class cMCML :public cCUDAMCML{
	CString m_cstrOutputName;
	CString m_tmpStrData;
	unsigned int m_nNumStr;
	float m_fVer;
	CStdioFile  cOutputFile;
	CString CreateSimRsltFile();
	inline int WriteSimRslts(MemStruct* HostMem, SimulationStruct* sim);
	inline void WriteOPL(CStdioFile*  cOutputFile){
		short l;
		CString TransCstr;
		for (l = 1; l <= m_simulations->n_layers; l++)
		{
			//fprintf(file, "The %d layer\n", l);
			TransCstr.Format(_T("The %d layer\n"), l);
			cOutputFile->WriteString(TransCstr);
			// fprintf(file, "%12.4E\n",Out_Prm.opl[l]);	/* 平均光路長の書き込み */
			TransCstr.Format(_T("%12.4E\n"), m_sHostMem.Out_Ptr->opl[l]);
			cOutputFile->WriteString(TransCstr);
			//fprintf(file, "\n");
			TransCstr.Format(_T("\n"));
			cOutputFile->WriteString(TransCstr);
		}
	}
	inline void WriteRd_ra(CStdioFile*  cOutputFile)
	{
		short it, ia;
		CString TransCstr;

		TransCstr.Format(_T("# Rd[theta][angle].[1 / (cm2sr)].\n"));
		cOutputFile->WriteString(TransCstr);
		TransCstr.Format(_T("# Rd[0][0],[0][1],..[0][na-1].\n"));
		cOutputFile->WriteString(TransCstr);
		TransCstr.Format(_T("# Rd[1][0],[1][1],..[1][na-1].\n"));
		cOutputFile->WriteString(TransCstr);
		TransCstr.Format(_T("# ...\n"));
		cOutputFile->WriteString(TransCstr);
		TransCstr.Format(_T("# Rd[nt-1][0],[nt-1][1],..[nt-1][na-1].\n"));
		cOutputFile->WriteString(TransCstr);
		TransCstr.Format(_T("# Rd_ta"));
		cOutputFile->WriteString(TransCstr);

		for (it = 0; it < m_simulations->det.nr; it++)
		{
			for (ia = 0; ia < m_simulations->det.na; ia++)
			{
				TransCstr.Format(_T("%12.4E,"), m_sHostMem.Out_Ptr->Rd_ra[it + ia*m_simulations->det.nr]);
				cOutputFile->WriteString(TransCstr);
				if ((it*m_simulations->det.na + ia + 1) % 9 == 0)
					TransCstr.Format(_T("\n"));
				cOutputFile->WriteString(TransCstr);
			}
		}
		TransCstr.Format(_T("\n"));
		cOutputFile->WriteString(TransCstr);
	}
	inline void WriteRd_p(CStdioFile*  cOutputFile)
	{
		short it, ia;
		CString TransCstr;
		TransCstr.Format(_T("# Rd[theta][angle].[1 / (cm2sr)].\n"));
		cOutputFile->WriteString(TransCstr);
		TransCstr.Format(_T("# Rd[0][0],[0][1],..[0][na-1].\n"));
		cOutputFile->WriteString(TransCstr);
		TransCstr.Format(_T("# Rd[1][0],[1][1],..[1][na-1].\n"));
		cOutputFile->WriteString(TransCstr);
		TransCstr.Format(_T("# ...\n"));
		cOutputFile->WriteString(TransCstr);
		TransCstr.Format(_T("# Rd[nt-1][0],[nt-1][1],..[nt-1][na-1].\n"));
		cOutputFile->WriteString(TransCstr);
		TransCstr.Format(_T("# Rd_p.\n"));
		cOutputFile->WriteString(TransCstr);

		for (it = 0; it < m_simulations->det.nr; it++)
		{
			for (ia = 0; ia < m_simulations->det.na; ia++)
			{
				TransCstr.Format(_T("%12.4E,"), m_sHostMem.Out_Ptr->Rd_p[it + ia*m_simulations->det.nr]);
				cOutputFile->WriteString(TransCstr);
				if ((it*m_simulations->det.na + ia + 1) % 9 == 0)
					TransCstr.Format(_T("\n"));
				cOutputFile->WriteString(TransCstr);
			}
		}
		TransCstr.Format(_T("photon number"));
		cOutputFile->WriteString(TransCstr);
		TransCstr.Format(_T("%ld\n"), m_sHostMem.Out_Ptr->p1);
		cOutputFile->WriteString(TransCstr);
		TransCstr.Format(_T("\n"));
		cOutputFile->WriteString(TransCstr);
	}
	inline void WriteInParm(CStdioFile*  cOutputFile, SimulationStruct sim)
	{
		CString TransCstr;
		short i;
		TransCstr.Format(_T("InParm \t\t\t# Input parameters. cm is used.\n"));
		cOutputFile->WriteString(TransCstr);
		
		TransCstr.Format(_T("%s \tA\t\t# output file name, ASCII.\n"),sim.outp_filename);
		cOutputFile->WriteString(TransCstr);

		TransCstr.Format(_T("%ld \t\t\t# No. of photons\n"), sim.number_of_photons);
		cOutputFile->WriteString(TransCstr);
		
		TransCstr.Format(_T("%.2lf \t\t\t# No. of SD distance\n"), sim.r);
		cOutputFile->WriteString(TransCstr);
		TransCstr.Format(_T("%G\t\t\t\t# dt [cm]\n"), sim.dt);
		cOutputFile->WriteString(TransCstr);

		TransCstr.Format(_T("%hd\t%hd\t\t# No. of dt, da.\n\n"),sim.nr, sim.na);
		cOutputFile->WriteString(TransCstr);
		
		TransCstr.Format(_T("%hd\t\t\t\t\t# Number of layers\n"),sim.n_layers);
		cOutputFile->WriteString(TransCstr);
		
		TransCstr.Format(_T("#n\tmua\tmus\tg\td\t# One line for each layer\n"));
		cOutputFile->WriteString(TransCstr);
		
		TransCstr.Format(_T("%G\t\t\t\t\t# n for medium above\n"),sim.layers[0].n);
		cOutputFile->WriteString(TransCstr);
		
		for (i = 1; i <= sim.n_layers; i++)  {
			LayerStruct s;
			s = sim.layers[i];
			
			TransCstr.Format(_T("%G\t%G\t%G\t%G\t%G\t# layer %hd\n"),s.n, s.mua, s.mutr, s.g, s.z_max - s.z_min, i);
			cOutputFile->WriteString(TransCstr);

			
		}
		TransCstr.Format(_T("%G\t\t\t\t\t# n for medium below\n\n"),sim.layers[i].n);
		cOutputFile->WriteString(TransCstr);
		
	}

	inline bool ReadStringRdy(CStdioFile* cFile){
		if (m_nNumStr == 0){
			if (!cFile->ReadString(m_tmpStrData)){
				return FALSE;
			}
			
		}
		return TRUE;
	}
	inline bool ReadFile1LineFlt(CStdioFile* cOpenedFile, float* RetDATA){
		while (ReadStringRdy(cOpenedFile)){
			unsigned int ntmpStartStr = m_nNumStr;	// 開始文字位置（タブ，スペース数)
			while (m_tmpStrData.Mid(ntmpStartStr, 1) == " "
				|| m_tmpStrData.Mid(ntmpStartStr, 1) == "\t"
				){
				ntmpStartStr++;
			}
			unsigned int ntmpEndStr = ntmpStartStr;	// 終端文字位置
			while (m_tmpStrData.Mid(ntmpEndStr, 1) != "#"
				&& m_tmpStrData.Mid(ntmpEndStr, 1) != ","
				&& m_tmpStrData.Mid(ntmpEndStr, 1) != " "
				&& m_tmpStrData.Mid(ntmpEndStr, 1) != "\t"
				){
				ntmpEndStr++;
			}
			if (ntmpEndStr != ntmpStartStr){
				if (m_tmpStrData.Mid(ntmpEndStr, 1) != "#"
					){
					*RetDATA = _tcstof(m_tmpStrData.Mid(ntmpStartStr, ntmpEndStr - ntmpStartStr), NULL);
					m_nNumStr = ntmpEndStr + 1;
				}
				else{
					*RetDATA = _tcstof(m_tmpStrData.Mid(ntmpStartStr, ntmpEndStr - ntmpStartStr), NULL);
					m_nNumStr = 0;
				}
				return TRUE;
			}
			m_nNumStr = 0;

		}
		return FALSE;
	}
	inline bool ReadFile1LineCStr(CStdioFile* cOpenedFile, CString* RetDATA){
		while (ReadStringRdy(cOpenedFile)){
			unsigned int ntmpStartStr = m_nNumStr;	// 開始文字位置（タブ，スペース数)
			while (m_tmpStrData.Mid(ntmpStartStr, 1) == " " 
				|| m_tmpStrData.Mid(ntmpStartStr, 1) == "\t"
				){
				ntmpStartStr++;
			}
			unsigned int ntmpEndStr = ntmpStartStr;	// 終端文字位置
			while (m_tmpStrData.Mid(ntmpEndStr, 1)	!= "#"
				&& m_tmpStrData.Mid(ntmpEndStr, 1)	!= ","
				&& m_tmpStrData.Mid(ntmpEndStr, 1)	!= " "
				&& m_tmpStrData.Mid(ntmpEndStr, 1)	!= "\t"
				){
				ntmpEndStr++;
				}
			if (ntmpEndStr != ntmpStartStr){
				if (m_tmpStrData.Mid(ntmpEndStr, 1) != "#"
					){
					*RetDATA = m_tmpStrData.Mid(ntmpStartStr, ntmpEndStr - ntmpStartStr);
					m_nNumStr = ntmpEndStr + 1;
				}
				else{
					*RetDATA = m_tmpStrData.Mid(ntmpStartStr, ntmpEndStr - ntmpStartStr);
					m_nNumStr = 0;
				}
				return TRUE;
			}
			m_nNumStr = 0;

		}
		return FALSE;
	}
	inline bool ReadFile1LineUInt(CStdioFile* cOpenedFile, unsigned int* RetDATA){
		while (ReadStringRdy(cOpenedFile)){
			unsigned int ntmpStartStr = m_nNumStr;	// 開始文字位置（タブ，スペース数)
			while (m_tmpStrData.Mid(ntmpStartStr, 1) == " "
				|| m_tmpStrData.Mid(ntmpStartStr, 1) == "\t"
				){
				ntmpStartStr++;
			}
			unsigned int ntmpEndStr = ntmpStartStr;	// 終端文字位置
			while (m_tmpStrData.Mid(ntmpEndStr, 1) != "#"
				&& m_tmpStrData.Mid(ntmpEndStr, 1) != ","
				&& m_tmpStrData.Mid(ntmpEndStr, 1) != " "
				&& m_tmpStrData.Mid(ntmpEndStr, 1) != "\t"
				){
				ntmpEndStr++;
			}
			if (ntmpEndStr != ntmpStartStr){
				if (m_tmpStrData.Mid(ntmpEndStr, 1) != "#"
					){
					*RetDATA = _tcstoul(m_tmpStrData.Mid(ntmpStartStr, ntmpEndStr - ntmpStartStr), NULL, 10);
					m_nNumStr = ntmpEndStr + 1;
				}
				else{
					*RetDATA = _tcstoul(m_tmpStrData.Mid(ntmpStartStr, ntmpEndStr - ntmpStartStr), NULL, 10);
					m_nNumStr = 0;
				}
				return TRUE;
			}
			m_nNumStr = 0;
		}
		return FALSE;
	}
	inline bool ReadFile1LineInt(CStdioFile* cOpenedFile, int* RetDATA){
		while (ReadStringRdy(cOpenedFile)){
			unsigned int ntmpStartStr = m_nNumStr;	// 開始文字位置（タブ，スペース数)
			while (m_tmpStrData.Mid(ntmpStartStr, 1) == " "
				|| m_tmpStrData.Mid(ntmpStartStr, 1) == "\t"
				){
				ntmpStartStr++;
			}
			unsigned int ntmpEndStr = ntmpStartStr;	// 終端文字位置
			while (m_tmpStrData.Mid(ntmpEndStr, 1) != "#"
				&& m_tmpStrData.Mid(ntmpEndStr, 1) != ","
				&& m_tmpStrData.Mid(ntmpEndStr, 1) != " "
				&& m_tmpStrData.Mid(ntmpEndStr, 1) != "\t"
				){
				ntmpEndStr++;
			}
			if (ntmpEndStr != ntmpStartStr){
				if (m_tmpStrData.Mid(ntmpEndStr, 1) != "#"
					){
					*RetDATA = _tcstol(m_tmpStrData.Mid(ntmpStartStr, ntmpEndStr - ntmpStartStr), NULL, 10);
					m_nNumStr = ntmpEndStr + 1;
				}
				else{
					*RetDATA = _tcstol(m_tmpStrData.Mid(ntmpStartStr, ntmpEndStr - ntmpStartStr), NULL, 10);
					m_nNumStr = 0;
				}
				return TRUE;
			}
			m_nNumStr = 0;
		}
		return FALSE;
	}
	inline bool ReadFile1LineUInt64(CStdioFile* cOpenedFile, unsigned long long* RetDATA){
		while (ReadStringRdy(cOpenedFile)){
			unsigned int ntmpStartStr = m_nNumStr;	// 開始文字位置（タブ，スペース数)
			while (m_tmpStrData.Mid(ntmpStartStr, 1) == " "
				|| m_tmpStrData.Mid(ntmpStartStr, 1) == "\t"
				){
				ntmpStartStr++;
			}
			unsigned int ntmpEndStr = ntmpStartStr;	// 終端文字位置
			while (m_tmpStrData.Mid(ntmpEndStr, 1) != "#"
				&& m_tmpStrData.Mid(ntmpEndStr, 1) != ","
				&& m_tmpStrData.Mid(ntmpEndStr, 1) != " "
				&& m_tmpStrData.Mid(ntmpEndStr, 1) != "\t"
				){
				ntmpEndStr++;
			}
			if (ntmpEndStr != ntmpStartStr){
				if (m_tmpStrData.Mid(ntmpEndStr, 1) != "#"
					){
					*RetDATA = _tcstoui64(m_tmpStrData.Mid(ntmpStartStr, ntmpEndStr - ntmpStartStr), NULL, 10);
					m_nNumStr = ntmpEndStr + 1;
				}
				else{
					*RetDATA = _tcstoui64(m_tmpStrData.Mid(ntmpStartStr, ntmpEndStr - ntmpStartStr), NULL, 10);
					m_nNumStr = 0;
				}
				return TRUE;
			}
			m_nNumStr = 0;
		}
		return FALSE;
	}
	int InitRNG( const unsigned long long n_rng,CString *safeprimes_file, unsigned long long xinit);
	CString ReadSimData(CString* filename, SimulationStruct** simulations, int ignoreAdetection,unsigned int Seed);



public:
	void InitIO();
	CString StartSim(CString* chPathName, int nPathName, CString* cstrB32Name, int B32NameLeng,unsigned int Seed);
};
