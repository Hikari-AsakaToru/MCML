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
	CStdioFile* m_lpcOutPutFile;
	
	CString CreateSimRsltFile();
	inline int WriteSimRslts(MemStruct* HostMem, SimulationStruct* sim);

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
