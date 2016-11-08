#include "afxdialogex.h"
#include "stdafx.h"
class FileIO{
	CString m_tmpStrData;
	unsigned int m_nNumStr;
	CStdioFile* cOpenedFile;
	inline bool ReadStringRdy(CStdioFile* cFile){
		if (m_nNumStr == 0){
			if (!cFile->ReadString(m_tmpStrData)){
				return FALSE;
			}

		}
		return TRUE;
	}
public:
	inline bool ReadFile1Flt( float* RetDATA){
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
				&& m_tmpStrData.Mid(ntmpEndStr, 1) != "\0"
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
			else{
				if (m_tmpStrData.Mid(ntmpEndStr, 1) == "#"
					|| m_tmpStrData.Mid(ntmpEndStr, 1) == "\0"
					){
					m_nNumStr = 0;
				}
				else{
					m_nNumStr++;
				}
			}

		}
		return FALSE;
	}
	inline bool ReadFile1CStr(CString* RetDATA){
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
				&& m_tmpStrData.Mid(ntmpEndStr, 1) != "\0"
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
			else{
				if (m_tmpStrData.Mid(ntmpEndStr, 1) == "#"
					|| m_tmpStrData.Mid(ntmpEndStr, 1) == "\0"
					){
					m_nNumStr = 0;
				}
				else{
					m_nNumStr++;
				}
			}

		}
		return FALSE;
	}
	inline bool ReadFile1UInt( unsigned int* RetDATA){
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
				&& m_tmpStrData.Mid(ntmpEndStr, 1) != "\0"
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
			else{
				if (m_tmpStrData.Mid(ntmpEndStr, 1) == "#"
					|| m_tmpStrData.Mid(ntmpEndStr, 1) == "\0"
					){
					m_nNumStr = 0;
				}
				else{
					m_nNumStr++;
				}
			}
		}
		return FALSE;
	}
	inline bool ReadFile1Int( int* RetDATA){
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
				&& m_tmpStrData.Mid(ntmpEndStr, 1) != "\0"
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
			else{
				if (m_tmpStrData.Mid(ntmpEndStr, 1) == "#"
					|| m_tmpStrData.Mid(ntmpEndStr, 1) == "\0"
					){
					m_nNumStr = 0;
				}
				else{
					m_nNumStr++;
				}
			}
		}
		return FALSE;
	}
	inline bool ReadFile1UInt64(unsigned long long* RetDATA){
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
			else{
				if (m_tmpStrData.Mid(ntmpEndStr, 1) == "#"
					|| m_tmpStrData.Mid(ntmpEndStr, 1) == "\0"
					){
					m_nNumStr = 0;
				}
				else{
					m_nNumStr++;
				}
			}
		}
		return FALSE;
	}
	inline void SetFile(CStdioFile* cInputFile){
		cOpenedFile = cInputFile;
	}

};