// $Id$
// -*-C++-*-
// * BeginRiceCopyright *****************************************************
// 
// Copyright ((c)) 2002, Rice University 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// 
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of Rice University (RICE) nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
// 
// This software is provided by RICE and contributors "as is" and any
// express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular
// purpose are disclaimed. In no event shall RICE or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or
// business interruption) however caused and on any theory of liability,
// whether in contract, strict liability, or tort (including negligence
// or otherwise) arising in any way out of the use of this software, even
// if advised of the possibility of such damage. 
// 
// ******************************************************* EndRiceCopyright *

#ifndef HPCViewSAX2_h
#define HPCViewSAX2_h

//************************ System Include Files ******************************

//************************* Xerces Include Files *****************************

#include <xercesc/sax2/SAX2XMLReader.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>
using XERCES_CPP_NAMESPACE::SAX2XMLReader;
using XERCES_CPP_NAMESPACE::SAX2XMLReader;
using XERCES_CPP_NAMESPACE::SAX2XMLReader;
using XERCES_CPP_NAMESPACE::XMLReaderFactory;
using XERCES_CPP_NAMESPACE::XMLUni;
using XERCES_CPP_NAMESPACE::SAXException;

#include <xercesc/sax2/DefaultHandler.hpp>
using XERCES_CPP_NAMESPACE::DefaultHandler;
using XERCES_CPP_NAMESPACE::Attributes;

#include <xercesc/sax/SAXParseException.hpp>
using XERCES_CPP_NAMESPACE::SAXParseException;

#include <xercesc/sax/ErrorHandler.hpp>
using XERCES_CPP_NAMESPACE::ErrorHandler;

//************************* User Include Files *******************************

# include <lib/support/String.h>

//************************ Forward Declarations ******************************

extern String getAttr(const Attributes& attributes, int iU); 
extern String getAttr(const Attributes& attributes, const XMLCh* const attr); 

#endif
