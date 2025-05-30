#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include "GaussLegendre2D.hpp"
#include <Eigen/Dense>
#include "activity.h"
#include "bilinear_interpol.h"

using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::MatrixXd;

using namespace std;

long double sph_norm(const long double theta, const long double phi, const int l, const int m){
	long double Re, Im;
	Re=boost::math::spherical_harmonic_r(l, m, theta, phi);
    Im=boost::math::spherical_harmonic_i(l, m, theta, phi);
	return (Re*Re + Im*Im)*std::sin(theta);
}


long double sph_norm2(const long double theta, const long double phi, const int l, const int m, const int dummy1, const int dummy2){
	long double Re, Im;
	Re=boost::math::spherical_harmonic_r(l, m, theta, phi);
    Im=boost::math::spherical_harmonic_i(l, m, theta, phi);
	return (Re*Re + Im*Im)*std::sin(theta);
}

//long double Alm_norm_gate(const double theta, const double phi, const int l, const int m, const long double theta0, const long double delta){
long double Alm_norm_gate(const long double theta, const long double phi, const int l, const int m, const long double theta0, const long double delta){
	long double sph, Fmax;
  	VectorXd tmp(1),F;
	sph=sph_norm(theta, phi, l,m);
  	tmp[0]=theta;
    F=gate_filter(tmp, theta0, delta);
    return sph*F[0];
}

long double Alm_norm_gauss(const long double theta, const long double phi, const int l, const int m, const long double theta0, const long double delta){
	long double sph, Fmax;
	VectorXd tmp(1),F;
	sph=sph_norm(theta, phi, l, m);
	tmp[0]=theta;
	F=gauss_filter(tmp, theta0, delta);
	return sph*F[0]; // Note on the normalisation: The normalisation using gauss_filter_cte() happens in main.cpp for optimisation purpose 

}

long double Alm_norm_triangle(const long double theta, const long double phi, const int l, const int m, const long double theta0, const long double delta){
	long double sph, Fmax;
	VectorXd tmp(1),F;
	sph=sph_norm(theta, phi, l, m);
	tmp[0]=theta;
	F=triangle_filter(tmp, theta0, delta);

	return sph*F[0]; 
}

// A funcion used to perform a Gaussian filtering on the integral term for the 
// 
VectorXd gauss_filter(const VectorXd theta, const long double theta0, const long double delta){
	long double a0;
	long double a1;
	VectorXd F(theta.size());
	for (int i=0; i<theta.size(); i++){
		a0=std::pow(theta[i] - theta0,2)/(2*std::pow(delta,2));
		a1=std::pow(theta[i] - M_PI + theta0, 2)/(2*std::pow(delta,2));	
		F[i]=std::exp(-a0) + std::exp(-a1);
	} 
	return F;
}

VectorXd gate_filter(const VectorXd theta, const long double theta0, const long double delta){
	VectorXd F(theta.size());
	F.setZero();
	const long double distance_critic=0.4;
	long double distance=M_PI - 2*theta0 - delta;
	for (int i=0; i<theta.size(); i++){
		if ((theta[i] >= (theta0 - delta/2) && theta[i] <= (theta0 + delta/2)) || 
		   (theta[i] >= (M_PI - theta0 - delta/2) && theta[i] <= (M_PI - theta0 + delta/2)) ) {
		   	F[i]=1;
		   }
	}
	return F;
}

VectorXd triangle_filter(const VectorXd theta, const long double theta0, const long double delta){
	/*
	A triangular filter which peaks at theta = theta0 and F=1
	Defined between theta = [0, pi/2]
	But I allow theta = [pi/2, pi] as well, TO BE USED FOR VISUALISATION ONLY
	For this model to make sense, note that theta0 must be in [0,pi/2]
	*/
	long double a, b;
	VectorXd F(theta.size());
	F.setZero();
	//
	// Lower part: Between 0<theta<theta0
	a=2/delta; // The slope
	b=1- a*theta0;
	for (int i=0; i<theta.size(); i++){
		if ( (theta[i] <= theta0) && (theta[i] >=0) && (theta[i] - (theta0 - delta/2)) >0){
			F[i]=a*theta[i] + b;
		}
	}
	// Higher part: Between theta0<theta<pi/2
	a=-2/delta; // The slope
	b=1- a*theta0;
	for (int i=0; i<theta.size(); i++){
		if ( (theta[i] > theta0) && (theta[i] <= M_PI/2) && (theta[i] - (theta0 + delta/2)) < 0){
			F[i]=a*theta[i] + b;
		} 
	}
	if (theta.maxCoeff() > M_PI/2){
		// Lower part: Between 0<theta<theta0
		a=2/delta; // The slope
		b=1- a*(M_PI - theta0);
		for (int i=0; i<theta.size(); i++){
			if ((theta[i] - (M_PI - theta0-delta/2)) > 0 && theta[i] <= M_PI-theta0  && theta[i] >= M_PI/2){
				F[i]=a*theta[i] + b;
			}
		}
		// Higher part: Between theta0<theta<np.pi/2
		a=-2/delta;
		b=1-a*(M_PI - theta0 );
		for (int i=0; i<theta.size(); i++){
			if ((theta[i]- (M_PI - theta0 + delta/2)) < 0 && theta[i] > M_PI - theta0 && theta[i] <= M_PI){
				F[i]=a*theta[i] + b;
			}
		}
	}
	return F;
}

long double gauss_filter_cte(const long double theta0, const long double delta){
	/*
		A function that calculate what is the max value of a double-gaussian filter
		that is symetrical towards pi/2 and bounded between [0, pi] and of width delta
                     -                      -
		          -    -                  -   -
		         -       -              -      -
		        -         -            -        -
		       -            -         -          - 
		      -              -       -            -
		    -                  -   -                -
		  -                      -                    -
		--+----------+-----------+---------------------+----> theta
		  0         theta0      pi/2    pi-theta0      pi
	*/

	//VectorXd theta= linspace(0, M_PI, 200);
	VectorXd theta = Eigen::VectorXd::LinSpaced(200, 0, M_PI);
	VectorXd F;
	F=gauss_filter(theta, theta0, delta);
	return F.maxCoeff();
}


long double integrate_Alm_triangle(const int l, const int m, const long double theta0, const long double delta){
	_2D::GQ::GaussLegendreQuadrature<double,64> integrate;
	long double theta_min=theta0-delta/2; // The range is maximum this. Above/Below it is just 0
	long double theta_max=theta0+delta/2;
	const long double phi_min=0;
	const long double phi_max=2*M_PI;
	const int dummy =0;
	long double r;
	if (theta_min < 0){ // We need to truncate the triangular profile to avoid getting out of the definition range: [0, PI/2]
		theta_min=0;
	}
	if (theta_max > M_PI/2){  // We need to truncate the triangular profile to avoid getting out of the definition range: [0, PI/2]
		theta_max=M_PI/2;
	}
	if (delta != 0){
			r = integrate(Alm_norm_triangle, theta_min, theta_max, phi_min, phi_max, l, m, theta0, delta);
	} else{
		r=0; // When delta is 0, obviously the result is 0
	}
	return r;
}


long double integrate_Alm_gate(const int l, const int m, const long double theta0, const long double delta){
	_2D::GQ::GaussLegendreQuadrature<double,64> integrate;
	long double theta_min=theta0-delta/2; // Default for ftype='gate'
	long double theta_max=theta0+delta/2;
	const long double phi_min=0;
	const long double phi_max=2.*M_PI;
	const int dummy =0;
	long double r;
	if (theta_min < 0){ // We need to truncate the triangular profile to avoid getting out of the definition range: [0, PI/2]
		theta_min=0;
	}
	if (theta_max > M_PI/2){  // We need to truncate the triangular profile to avoid getting out of the definition range: [0, PI/2]
		theta_max=M_PI/2;
	}
	if (delta != 0){
			r = integrate(sph_norm2, theta_min, theta_max, phi_min, phi_max, l, m, dummy, dummy);
	} else{
		r=0; // When delta is 0, obviously the result is 0
	}
	return r;
}

long double Alm(const int l, const int m, const long double theta0, const long double delta, std::string ftype){
	_2D::GQ::GaussLegendreQuadrature<double,64> integrate;
	//_2D::GQ::GaussLegendreQuadrature<double,16> integrate;
	const long double theta_min=0; // Default for ftype='gate'
	const long double theta_max=M_PI/2;
	const long double phi_min=0;
	const long double phi_max=2.*M_PI;

	long double r;
	if (std::abs(m)<=l){
		if (ftype == "gate"){
			r=integrate_Alm_gate(l, m, theta0, delta); //A function that integrate directly using Ylm (no filtering, which is faster)
		}
		if (ftype == "gauss"){
			r=integrate(Alm_norm_gauss, theta_min, theta_max, phi_min, phi_max, l, m, theta0, delta);
		}
		if (ftype == "triangle"){
			r=integrate_Alm_triangle(l, m, theta0, delta);
		}		
		r=2*r; // Accouting for the North + South Emisphere
	} else{
		r=-10;
		std::cout << "Alm Error: -l<m<l not respected. Will return -10" << std::endl;
	}
	return r;
}

long double Alm_deg(const int l, const int m, const long double theta0, const long double delta, std::string ftype){

	_2D::GQ::GaussLegendreQuadrature<double,64> integrate;
	const long double theta_min=0; // Default for ftype='gate'
	const long double theta_max=M_PI/2;
	const long double phi_min=0;
	const long double phi_max=2.*M_PI;

	long double r;
	if (std::abs(m)<=l){
		if (ftype == "gate"){
			r=integrate_Alm_gate(l, m, theta0*M_PI/180., delta*M_PI/180.); //A function that integrate directly using Ylm (no filtering, which is faster)
		}
		if (ftype == "gauss"){
			r=integrate(Alm_norm_gauss, theta_min, theta_max, phi_min, phi_max, l, m, theta0*M_PI/180., delta*M_PI/180.);
		}
		if (ftype == "triangle"){
			r=integrate_Alm_triangle(l, m, theta0*M_PI/180., delta*M_PI/180.); //A function that integrate directly using Ylm (no filtering, which is faster)
		}
		r=2*r; // Accouting for the North + South Emisphere
	} else{
		r=-10;
		std::cout << "Alm Error: -l<m<l not respected. Will return -10" << std::endl;
	}
	return r;
}
