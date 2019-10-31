#pragma once

#include "meshlab/Mesh.h"

#include <set>

namespace meshlab
{

class Igarashi05
{
public:
	Igarashi05(const std::vector<vec2f>& verts, const std::vector<size_t>& tris);

	void ResetConstraints(const std::vector<int>& constraints);

	void UpdateMesh(bool rigid);

private:
	struct Triangle
	{
		size_t idx_verts[3];

		// definition of each vertex in triangle-local coordinate system
		vec2f  tri_coords[3];

		// un-scaled triangle
		vec2f  scaled[3];

		// pre-computed matrices for triangle scaling step
		Eigen::MatrixXd F, C;
	};

private:
	void BuidTriangles(const std::vector<size_t>& tris);

	// 4.1 Step one : scale - free construction
	void PrecomputeOrientationMatrix();
	// 4.2.1 Fitting the original triangle to the intermediate triangle
	void PrecomputeScalingMatrices(Triangle& tri);
	// 4.2.2 Generating the final result using the fitted triangles
	void PrecomputeFittingMatrices();

	// 4.2.1
	void UpdateScaledTriangle(Triangle& tri);
	// 4.2.2
	void ApplyFittingStep();

private:
	// init
	std::vector<vec2f>    m_verts;
	std::vector<Triangle> m_tris;

	// deform
	std::vector<vec2f> m_deformed;

	// constraints
	std::set<size_t> m_constraints;

	// these three matrix should be precompute according to Eq(6)-(8)
	Eigen::MatrixXd m_G;
	Eigen::MatrixXd m_Gprime;
	Eigen::MatrixXd m_B;

	Eigen::MatrixXd m_first_mat;

	Eigen::MatrixXd m_HXPrime, m_HYPrime;
	Eigen::MatrixXd m_DX, m_DY;
	Eigen::MatrixXd m_LUDecompX, m_LUDecompY;

	// reorder the verts, the free vertices are put in the front
	// and the control vertices are put in the back
	std::vector<size_t> m_vertex_map;

}; // Igarashi05

}