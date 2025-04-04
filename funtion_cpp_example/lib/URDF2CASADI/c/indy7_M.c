/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) indy7_M_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s1[45] = {6, 6, 0, 6, 12, 18, 24, 30, 36, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};

/* M:(i0[6])->(o0[6x6]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a7, a8, a9;
  a0=5.9644150000000000e-02;
  a1=2.2204460492503131e-16;
  a2=arg[0]? arg[0][1] : 0;
  a3=sin(a2);
  a4=(a1*a3);
  a2=cos(a2);
  a4=(a4-a2);
  a5=2.9356979999999994e-01;
  a6=3.4245930000000001e-02;
  a7=6.7040499999999996e-03;
  a8=9.9489100000000000e-03;
  a9=-3.1185000000000002e-04;
  a10=-1.6100000000000001e-04;
  a11=arg[0]? arg[0][5] : 0;
  a12=cos(a11);
  a13=(a10*a12);
  a14=-8.8817841970012525e-24;
  a15=1.3000000000000000e-07;
  a11=sin(a11);
  a16=(a15*a11);
  a16=(a14-a16);
  a13=(a13+a16);
  a16=(a13*a12);
  a17=(a15*a12);
  a18=2.2648549702353193e-22;
  a19=-1.5085000000000001e-04;
  a20=(a19*a11);
  a20=(a18-a20);
  a17=(a17+a20);
  a20=(a17*a11);
  a16=(a16-a20);
  a9=(a9-a16);
  a20=(a15*a12);
  a14=(a14-a20);
  a10=(a10*a11);
  a14=(a14-a10);
  a10=(a14*a11);
  a19=(a19*a12);
  a18=(a18-a19);
  a15=(a15*a11);
  a18=(a18-a15);
  a15=(a18*a12);
  a10=(a10+a15);
  a9=(a9+a10);
  a15=5.9634000000000000e-04;
  a9=(a9+a15);
  a8=(a8+a9);
  a9=1.6239698144157814e-03;
  a8=(a8+a9);
  a9=2.7149200000000000e-03;
  a10=(a15-a10);
  a9=(a9+a10);
  a10=9.6271632098447833e-03;
  a9=(a9+a10);
  a8=(a8-a9);
  a10=9.7818899999999997e-03;
  a16=(a16+a15);
  a10=(a10+a16);
  a16=1.1251133024260564e-02;
  a10=(a10+a16);
  a10=(a10-a9);
  a16=(a8+a10);
  a19=arg[0]? arg[0][4] : 0;
  a20=sin(a19);
  a21=(a1*a20);
  a22=(a21*a8);
  a19=cos(a19);
  a23=(a1*a19);
  a24=1.4000000000000001e-07;
  a25=(a1*a12);
  a26=-2.2204460492503131e-16;
  a27=(a26*a11);
  a25=(a25+a27);
  a13=(a13*a25);
  a27=(a26*a12);
  a28=(a1*a11);
  a27=(a27-a28);
  a17=(a17*a27);
  a13=(a13+a17);
  a17=5.0999999999999999e-07;
  a28=(a17*a11);
  a29=-2.0000000000000000e-08;
  a30=(a29*a12);
  a28=(a28-a30);
  a13=(a13+a28);
  a24=(a24+a13);
  a13=(a23*a24);
  a28=2.;
  a30=3.2100000000000002e-06;
  a31=(a14*a25);
  a32=(a18*a27);
  a31=(a31+a32);
  a32=(a17*a12);
  a33=(a29*a11);
  a32=(a32+a33);
  a31=(a31+a32);
  a30=(a30+a31);
  a31=-3.9540134611862499e-03;
  a30=(a30-a31);
  a31=(a28*a30);
  a13=(a13-a31);
  a22=(a22+a13);
  a13=(a22*a21);
  a31=(a21*a24);
  a32=(a23*a10);
  a33=-9.3546000000000005e-04;
  a14=(a14*a12);
  a18=(a18*a11);
  a14=(a14-a18);
  a18=(a29*a27);
  a34=(a17*a25);
  a18=(a18-a34);
  a14=(a14+a18);
  a33=(a33+a14);
  a14=(a33+a33);
  a32=(a32-a14);
  a31=(a31+a32);
  a32=(a31*a23);
  a13=(a13+a32);
  a16=(a16-a13);
  a32=(a1*a20);
  a32=(a32-a19);
  a8=(a32*a8);
  a14=(a1*a19);
  a14=(a20+a14);
  a18=(a14*a24);
  a8=(a8+a18);
  a18=(a8*a32);
  a24=(a32*a24);
  a10=(a14*a10);
  a24=(a24+a10);
  a10=(a24*a14);
  a18=(a18+a10);
  a16=(a16-a18);
  a16=(a16+a9);
  a7=(a7+a16);
  a16=1.1865049876044926e+00;
  a10=-1.1400000000000000e-01;
  a34=-2.5038568645553184e-02;
  a35=(a34*a21);
  a36=-1.0283697836566486e-02;
  a35=(a35+a36);
  a35=(a10+a35);
  a36=casadi_sq(a35);
  a37=8.3000000000000004e-02;
  a38=(a34*a32);
  a38=(a37+a38);
  a39=casadi_sq(a38);
  a36=(a36+a39);
  a36=(a16*a36);
  a7=(a7+a36);
  a36=6.1934099999999999e-03;
  a18=(a18+a9);
  a36=(a36+a18);
  a18=(a1*a19);
  a18=(a18+a20);
  a34=(a34*a18);
  a39=2.2834396232888045e-18;
  a34=(a34+a39);
  a39=casadi_sq(a34);
  a40=casadi_sq(a35);
  a39=(a39+a40);
  a39=(a16*a39);
  a36=(a36+a39);
  a7=(a7-a36);
  a39=2.7924600000000001e-03;
  a13=(a13+a9);
  a39=(a39+a13);
  a13=casadi_sq(a34);
  a40=casadi_sq(a38);
  a13=(a13+a40);
  a13=(a16*a13);
  a39=(a39+a13);
  a39=(a39-a36);
  a13=(a7+a39);
  a40=arg[0]? arg[0][3] : 0;
  a41=cos(a40);
  a42=(a41*a7);
  a43=1.5000000000000000e-06;
  a44=(a8*a18);
  a20=(a1*a20);
  a19=(a19-a20);
  a20=(a24*a19);
  a44=(a44+a20);
  a20=(a23*a30);
  a45=(a21*a33);
  a20=(a20-a45);
  a44=(a44-a20);
  a43=(a43+a44);
  a44=(a16*a34);
  a44=(a44*a38);
  a43=(a43-a44);
  a44=(a28*a43);
  a20=(a1*a44);
  a40=sin(a40);
  a45=3.7500000000000001e-06;
  a22=(a22*a18);
  a31=(a31*a19);
  a22=(a22+a31);
  a31=(a14*a30);
  a46=(a32*a33);
  a31=(a31-a46);
  a22=(a22+a31);
  a45=(a45+a22);
  a22=(a16*a34);
  a22=(a22*a35);
  a45=(a45-a22);
  a22=(a40*a45);
  a20=(a20-a22);
  a42=(a42+a20);
  a20=(a42*a41);
  a22=(a41*a45);
  a31=-1.2796699999999999e-03;
  a8=(a8*a21);
  a24=(a24*a23);
  a8=(a8+a24);
  a24=(a19*a30);
  a46=(a18*a33);
  a24=(a24-a46);
  a8=(a8+a24);
  a31=(a31+a8);
  a16=(a16*a35);
  a16=(a16*a38);
  a31=(a31-a16);
  a16=(a31+a31);
  a8=(a1*a16);
  a24=(a40*a39);
  a8=(a8-a24);
  a22=(a22+a8);
  a8=(a22*a40);
  a20=(a20-a8);
  a13=(a13-a20);
  a44=(a1*a44);
  a8=(a41*a45);
  a44=(a44-a8);
  a7=(a40*a7);
  a44=(a44-a7);
  a7=(a44*a40);
  a16=(a1*a16);
  a39=(a41*a39);
  a16=(a16-a39);
  a45=(a40*a45);
  a16=(a16-a45);
  a45=(a16*a41);
  a7=(a7+a45);
  a13=(a13+a7);
  a13=(a13+a36);
  a6=(a6+a13);
  a13=1.8447343206269440e+00;
  a45=5.5883645304508534e-01;
  a34=(a45*a34);
  a39=(a41*a34);
  a38=(a45*a38);
  a8=(a1*a38);
  a45=(a45*a35);
  a35=(a40*a45);
  a8=(a8-a35);
  a39=(a39+a8);
  a8=casadi_sq(a39);
  a35=-7.4999999999999997e-02;
  a24=(a1*a38);
  a46=(a41*a45);
  a24=(a24-a46);
  a46=(a40*a34);
  a24=(a24-a46);
  a24=(a35+a24);
  a46=casadi_sq(a24);
  a8=(a8+a46);
  a8=(a13*a8);
  a6=(a6+a8);
  a8=4.5047700000000000e-03;
  a7=(a36-a7);
  a8=(a8+a7);
  a7=-2.6700000000000002e-01;
  a46=(a1*a41);
  a47=(a26*a40);
  a46=(a46+a47);
  a47=(a46*a34);
  a26=(a26*a41);
  a48=(a1*a40);
  a26=(a26-a48);
  a48=(a26*a45);
  a48=(a48-a38);
  a47=(a47+a48);
  a47=(a7+a47);
  a48=casadi_sq(a47);
  a49=casadi_sq(a39);
  a48=(a48+a49);
  a48=(a13*a48);
  a8=(a8+a48);
  a6=(a6-a8);
  a48=3.4060239999999999e-02;
  a20=(a20+a36);
  a48=(a48+a20);
  a20=casadi_sq(a47);
  a49=casadi_sq(a24);
  a20=(a20+a49);
  a20=(a13*a20);
  a48=(a48+a20);
  a48=(a48-a8);
  a20=(a6+a48);
  a49=arg[0]? arg[0][2] : 0;
  a50=sin(a49);
  a6=(a50*a6);
  a49=cos(a49);
  a51=1.4899999999999999e-06;
  a42=(a42*a46);
  a22=(a22*a26);
  a42=(a42+a22);
  a22=(a40*a31);
  a52=(a41*a43);
  a22=(a22-a52);
  a42=(a42+a22);
  a51=(a51+a42);
  a42=(a13*a47);
  a42=(a42*a39);
  a51=(a51-a42);
  a42=(a49*a51);
  a6=(a6+a42);
  a42=(a6*a50);
  a51=(a50*a51);
  a48=(a49*a48);
  a51=(a51+a48);
  a48=(a51*a49);
  a42=(a42+a48);
  a20=(a20-a42);
  a20=(a20+a8);
  a5=(a5+a20);
  a20=3.9486659444814474e+00;
  a48=6.1669136153995030e-01;
  a22=(a48*a47);
  a52=(a50*a22);
  a53=(a48*a39);
  a54=(a49*a53);
  a52=(a52+a54);
  a54=casadi_sq(a52);
  a55=-3.0499999999999999e-02;
  a48=(a48*a24);
  a56=(a55+a48);
  a57=casadi_sq(a56);
  a54=(a54+a57);
  a54=(a20*a54);
  a5=(a5+a54);
  a54=3.6206090000000003e-02;
  a54=(a54+a8);
  a57=-4.5000000000000001e-01;
  a58=(a49*a22);
  a59=(a50*a53);
  a58=(a58-a59);
  a58=(a57+a58);
  a59=casadi_sq(a58);
  a60=casadi_sq(a52);
  a59=(a59+a60);
  a59=(a20*a59);
  a54=(a54+a59);
  a5=(a5-a54);
  a5=(a4*a5);
  a59=(a1*a2);
  a59=(a3+a59);
  a60=-3.9999999999999998e-07;
  a6=(a6*a49);
  a51=(a51*a50);
  a6=(a6-a51);
  a60=(a60+a6);
  a6=(a20*a58);
  a6=(a6*a52);
  a60=(a60-a6);
  a6=(a59*a60);
  a5=(a5+a6);
  a5=(a5*a4);
  a60=(a4*a60);
  a6=2.8094142000000000e-01;
  a42=(a42+a8);
  a6=(a6+a42);
  a42=casadi_sq(a58);
  a51=casadi_sq(a56);
  a42=(a42+a51);
  a42=(a20*a42);
  a6=(a6+a42);
  a6=(a6-a54);
  a6=(a59*a6);
  a60=(a60+a6);
  a60=(a60*a59);
  a5=(a5+a60);
  a5=(a5+a54);
  a0=(a0+a5);
  a5=6.7554962018283531e+00;
  a60=(a1*a2);
  a60=(a60+a3);
  a6=4.9402036401124172e-01;
  a42=(a6*a58);
  a51=(a60*a42);
  a61=(a1*a3);
  a61=(a2-a61);
  a62=(a6*a52);
  a63=(a61*a62);
  a6=(a6*a56);
  a64=(a1*a6);
  a63=(a63+a64);
  a51=(a51+a63);
  a63=casadi_sq(a51);
  a64=-1.0900000000000000e-01;
  a3=(a1*a3);
  a3=(a3*a42);
  a2=(a1*a2);
  a2=(a2*a62);
  a2=(a2-a6);
  a3=(a3+a2);
  a3=(a64+a3);
  a2=casadi_sq(a3);
  a63=(a63+a2);
  a5=(a5*a63);
  a0=(a0+a5);
  a5=2.7599933319999998e+01;
  a63=5.7235366972980783e-01;
  a51=(a63*a51);
  a51=casadi_sq(a51);
  a63=(a63*a3);
  a63=casadi_sq(a63);
  a51=(a51+a63);
  a5=(a5*a51);
  a0=(a0+a5);
  if (res[0]!=0) res[0][0]=a0;
  a0=1.4409999999999999e-05;
  a5=7.2400000000000001e-06;
  a51=(a44*a46);
  a63=(a16*a26);
  a51=(a51+a63);
  a63=(a41*a31);
  a3=(a40*a43);
  a63=(a63+a3);
  a51=(a51+a63);
  a5=(a5+a51);
  a47=(a13*a47);
  a47=(a47*a24);
  a5=(a5-a47);
  a28=(a28*a5);
  a47=(a28*a49);
  a51=1.8600900000000000e-03;
  a44=(a44*a41);
  a16=(a16*a40);
  a44=(a44-a16);
  a16=(a26*a43);
  a63=(a46*a31);
  a16=(a16-a63);
  a44=(a44+a16);
  a51=(a51+a44);
  a13=(a13*a39);
  a13=(a13*a24);
  a51=(a51-a13);
  a13=(a51+a51);
  a24=(a13*a50);
  a47=(a47-a24);
  a24=(a49*a5);
  a39=(a50*a51);
  a24=(a24-a39);
  a47=(a47-a24);
  a0=(a0+a47);
  a58=(a20*a58);
  a58=(a58*a56);
  a0=(a0-a58);
  a58=1.5796923119999999e+01;
  a47=(a58*a42);
  a47=(a47*a6);
  a0=(a0-a47);
  a0=(a4*a0);
  a47=3.7279720000000002e-02;
  a28=(a28*a50);
  a13=(a13*a49);
  a28=(a28+a13);
  a13=(a49*a51);
  a24=(a50*a5);
  a13=(a13+a24);
  a28=(a28-a13);
  a47=(a47+a28);
  a20=(a20*a52);
  a20=(a20*a56);
  a47=(a47-a20);
  a20=(a58*a62);
  a20=(a20*a6);
  a47=(a47-a20);
  a47=(a59*a47);
  a0=(a0+a47);
  a47=-1.5796923119999999e+01;
  a47=(a47*a62);
  a47=(a60*a47);
  a20=(a58*a42);
  a20=(a61*a20);
  a47=(a47+a20);
  a47=(a64*a47);
  a0=(a0-a47);
  if (res[0]!=0) res[0][1]=a0;
  a47=7.8040017099999996e+00;
  a20=(a47*a22);
  a20=(a20*a48);
  a5=(a5-a20);
  a20=(a49*a5);
  a6=(a47*a53);
  a6=(a6*a48);
  a51=(a51-a6);
  a6=(a50*a51);
  a20=(a20-a6);
  a6=-7.8040017099999996e+00;
  a6=(a6*a53);
  a48=(a50*a6);
  a56=(a47*a22);
  a52=(a49*a56);
  a48=(a48+a52);
  a52=(a55*a48);
  a20=(a20-a52);
  a20=(a4*a20);
  a5=(a50*a5);
  a51=(a49*a51);
  a5=(a5+a51);
  a6=(a49*a6);
  a56=(a50*a56);
  a6=(a6-a56);
  a56=(a55*a6);
  a5=(a5+a56);
  a5=(a59*a5);
  a20=(a20+a5);
  a6=(a60*a6);
  a5=(a61*a48);
  a6=(a6+a5);
  a6=(a64*a6);
  a20=(a20-a6);
  if (res[0]!=0) res[0][2]=a20;
  a6=4.8126604400000002e+00;
  a5=(a6*a34);
  a5=(a5*a38);
  a43=(a43-a5);
  a5=(a46*a43);
  a56=(a6*a45);
  a56=(a56*a38);
  a31=(a31-a56);
  a56=(a26*a31);
  a38=casadi_sq(a34);
  a51=casadi_sq(a45);
  a38=(a38+a51);
  a38=(a6*a38);
  a36=(a36+a38);
  a56=(a56-a36);
  a5=(a5+a56);
  a56=-4.8126604400000002e+00;
  a56=(a56*a45);
  a45=(a41*a56);
  a6=(a6*a34);
  a34=(a40*a6);
  a45=(a45-a34);
  a34=(a35*a45);
  a5=(a5-a34);
  a34=(a49*a5);
  a38=(a41*a43);
  a51=(a1*a36);
  a52=(a40*a31);
  a51=(a51-a52);
  a38=(a38+a51);
  a51=(a46*a56);
  a52=(a26*a6);
  a51=(a51+a52);
  a52=(a35*a51);
  a56=(a40*a56);
  a6=(a41*a6);
  a56=(a56+a6);
  a6=(a7*a56);
  a52=(a52+a6);
  a38=(a38+a52);
  a52=(a50*a38);
  a34=(a34-a52);
  a52=(a50*a51);
  a6=(a49*a45);
  a52=(a52+a6);
  a6=(a55*a52);
  a34=(a34-a6);
  a34=(a4*a34);
  a5=(a50*a5);
  a38=(a49*a38);
  a5=(a5+a38);
  a51=(a49*a51);
  a38=(a50*a45);
  a51=(a51-a38);
  a38=(a55*a51);
  a6=(a57*a56);
  a38=(a38+a6);
  a5=(a5+a38);
  a5=(a59*a5);
  a34=(a34+a5);
  a51=(a60*a51);
  a5=(a61*a52);
  a56=(a1*a56);
  a5=(a5-a56);
  a51=(a51+a5);
  a51=(a64*a51);
  a34=(a34-a51);
  if (res[0]!=0) res[0][3]=a34;
  a51=-6.9251431337375032e-04;
  a30=(a30-a51);
  a51=(a18*a30);
  a5=(a19*a33);
  a56=1.6861218064752183e-03;
  a9=(a9+a56);
  a56=(a1*a9);
  a5=(a5+a56);
  a51=(a51+a5);
  a5=-6.7340982240000014e-02;
  a56=(a5*a14);
  a38=(a10*a56);
  a6=(a5*a23);
  a28=(a37*a6);
  a38=(a38-a28);
  a51=(a51+a38);
  a38=(a46*a51);
  a28=(a21*a30);
  a13=(a23*a33);
  a13=(a13-a9);
  a28=(a28+a13);
  a5=(a5*a19);
  a37=(a37*a5);
  a28=(a28+a37);
  a37=(a26*a28);
  a30=(a32*a30);
  a33=(a14*a33);
  a30=(a30+a33);
  a10=(a10*a5);
  a30=(a30-a10);
  a37=(a37-a30);
  a38=(a38+a37);
  a37=(a41*a5);
  a10=(a1*a56);
  a33=(a40*a6);
  a10=(a10-a33);
  a37=(a37+a10);
  a10=(a35*a37);
  a38=(a38-a10);
  a10=(a49*a38);
  a33=(a41*a51);
  a13=(a1*a30);
  a24=(a40*a28);
  a13=(a13-a24);
  a33=(a33+a13);
  a13=(a46*a5);
  a24=(a26*a6);
  a24=(a24-a56);
  a13=(a13+a24);
  a35=(a35*a13);
  a56=(a1*a56);
  a6=(a41*a6);
  a56=(a56-a6);
  a5=(a40*a5);
  a56=(a56-a5);
  a5=(a7*a56);
  a35=(a35-a5);
  a33=(a33+a35);
  a35=(a50*a33);
  a10=(a10-a35);
  a35=(a50*a13);
  a5=(a49*a37);
  a35=(a35+a5);
  a5=(a55*a35);
  a10=(a10-a5);
  a10=(a4*a10);
  a38=(a50*a38);
  a33=(a49*a33);
  a38=(a38+a33);
  a13=(a49*a13);
  a33=(a50*a37);
  a13=(a13-a33);
  a55=(a55*a13);
  a33=(a57*a56);
  a55=(a55-a33);
  a38=(a38+a55);
  a38=(a59*a38);
  a10=(a10+a38);
  a60=(a60*a13);
  a61=(a61*a35);
  a56=(a1*a56);
  a61=(a61+a56);
  a60=(a60+a61);
  a64=(a64*a60);
  a10=(a10-a64);
  if (res[0]!=0) res[0][4]=a10;
  a25=(a29*a25);
  a27=(a17*a27);
  a64=-5.9634000000000000e-04;
  a27=(a27+a64);
  a25=(a25+a27);
  a18=(a18*a25);
  a27=(a29*a12);
  a64=1.3241407970099317e-19;
  a60=(a17*a11);
  a60=(a64-a60);
  a27=(a27+a60);
  a19=(a19*a27);
  a17=(a17*a12);
  a64=(a64-a17);
  a29=(a29*a11);
  a64=(a64-a29);
  a29=(a1*a64);
  a19=(a19+a29);
  a18=(a18+a19);
  a46=(a46*a18);
  a21=(a21*a25);
  a23=(a23*a27);
  a23=(a23-a64);
  a21=(a21+a23);
  a26=(a26*a21);
  a32=(a32*a25);
  a14=(a14*a27);
  a32=(a32+a14);
  a26=(a26-a32);
  a46=(a46+a26);
  a26=(a49*a46);
  a14=(a41*a18);
  a27=(a1*a32);
  a25=(a40*a21);
  a27=(a27-a25);
  a14=(a14+a27);
  a27=(a50*a14);
  a26=(a26-a27);
  a4=(a4*a26);
  a50=(a50*a46);
  a49=(a49*a14);
  a50=(a50+a49);
  a59=(a59*a50);
  a4=(a4+a59);
  if (res[0]!=0) res[0][5]=a4;
  if (res[0]!=0) res[0][6]=a0;
  a42=casadi_sq(a42);
  a62=casadi_sq(a62);
  a42=(a42+a62);
  a58=(a58*a42);
  a54=(a54+a58);
  if (res[0]!=0) res[0][7]=a54;
  a22=casadi_sq(a22);
  a53=casadi_sq(a53);
  a22=(a22+a53);
  a47=(a47*a22);
  a8=(a8+a47);
  a48=(a57*a48);
  a48=(a8+a48);
  if (res[0]!=0) res[0][8]=a48;
  a47=(a1*a36);
  a31=(a41*a31);
  a47=(a47-a31);
  a43=(a40*a43);
  a47=(a47-a43);
  a45=(a7*a45);
  a47=(a47+a45);
  a52=(a57*a52);
  a52=(a47+a52);
  if (res[0]!=0) res[0][9]=a52;
  a45=(a1*a30);
  a28=(a41*a28);
  a45=(a45-a28);
  a51=(a40*a51);
  a45=(a45-a51);
  a7=(a7*a37);
  a45=(a45+a7);
  a57=(a57*a35);
  a57=(a45+a57);
  if (res[0]!=0) res[0][10]=a57;
  a1=(a1*a32);
  a41=(a41*a21);
  a1=(a1-a41);
  a40=(a40*a18);
  a1=(a1-a40);
  if (res[0]!=0) res[0][11]=a1;
  if (res[0]!=0) res[0][12]=a20;
  if (res[0]!=0) res[0][13]=a48;
  if (res[0]!=0) res[0][14]=a8;
  if (res[0]!=0) res[0][15]=a47;
  if (res[0]!=0) res[0][16]=a45;
  if (res[0]!=0) res[0][17]=a1;
  if (res[0]!=0) res[0][18]=a34;
  if (res[0]!=0) res[0][19]=a52;
  if (res[0]!=0) res[0][20]=a47;
  if (res[0]!=0) res[0][21]=a36;
  if (res[0]!=0) res[0][22]=a30;
  if (res[0]!=0) res[0][23]=a32;
  if (res[0]!=0) res[0][24]=a10;
  if (res[0]!=0) res[0][25]=a57;
  if (res[0]!=0) res[0][26]=a45;
  if (res[0]!=0) res[0][27]=a30;
  if (res[0]!=0) res[0][28]=a9;
  if (res[0]!=0) res[0][29]=a64;
  if (res[0]!=0) res[0][30]=a4;
  if (res[0]!=0) res[0][31]=a1;
  if (res[0]!=0) res[0][32]=a1;
  if (res[0]!=0) res[0][33]=a32;
  if (res[0]!=0) res[0][34]=a64;
  if (res[0]!=0) res[0][35]=a15;
  return 0;
}

CASADI_SYMBOL_EXPORT int M(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int M_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int M_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void M_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int M_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void M_release(int mem) {
}

CASADI_SYMBOL_EXPORT void M_incref(void) {
}

CASADI_SYMBOL_EXPORT void M_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int M_n_in(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_int M_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real M_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* M_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* M_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* M_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* M_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int M_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 1;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
