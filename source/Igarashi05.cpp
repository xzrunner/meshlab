// use code from: 
// https://github.com/zhangzhensong/arap
// http://www.dgp.toronto.edu/~rms/software/Deform2D/

#include "meshlab/Igarashi05.h"

namespace
{

void ExtractSubMatrix(Eigen::MatrixXd& mFrom, int nRowOffset, int nColOffset, Eigen::MatrixXd& mTo )
{
	int nRows = mTo.rows();
	int nCols = mTo.cols();

	for ( int i = 0; i < nRows; ++i ) {
		for ( int j = 0; j < nCols; ++j ) {
			mTo(i,j) = mFrom( i + nRowOffset, j + nColOffset );
		}
	}
}

}

namespace ml
{

Igarashi05::Igarashi05(const std::vector<vec2f>& verts, const std::vector<size_t>& tris)
	: m_verts(verts)
	, m_deformed(verts)
{
	BuidTriangles(tris);

	size_t sz = verts.size() * 2;
	m_G.resize(sz, sz);
//	m_G.setConstant(0);
}

void Igarashi05::ResetConstraints(const std::vector<int>& constraints)
{
	// update vertex map, reorder vertices
	for (auto& i : constraints) {
		m_constraints.insert(i);
	}
	size_t n_vert = m_verts.size();
	m_vertex_map.resize(n_vert);
	int row = 0;
	for (size_t i = 0; i < n_vert; ++i) {
		if (m_constraints.find(row) == m_constraints.end()) {
			m_vertex_map[i] = row++;
		}
	}
	for (auto& i : m_constraints) {
		m_vertex_map[i] = row++;
	}
	assert(row == n_vert);

	PrecomputeOrientationMatrix();
	for (auto& tri : m_tris) {
		PrecomputeScalingMatrices(tri);
	}
	PrecomputeFittingMatrices();
}

void Igarashi05::BuidTriangles(const std::vector<size_t>& tris)
{
	assert(tris.size() % 3 == 0);

	size_t n_tris = tris.size() / 3;
	m_tris.resize(n_tris);
	for (size_t i = 0; i < n_tris; ++i)
	{
		auto& t = m_tris[i];
		t.idx_verts[0] = tris[i * 3];
		t.idx_verts[1] = tris[i * 3 + 1];
		t.idx_verts[2] = tris[i * 3 + 2];
		for (size_t j = 0; j < 3; ++j)
		{
			size_t n0 = j;
			size_t n1 = (j + 1) % 3;
			size_t n2 = (j + 2) % 3;

			auto& v0 = m_verts[t.idx_verts[n0]];
			auto& v1 = m_verts[t.idx_verts[n1]];
			auto& v2 = m_verts[t.idx_verts[n2]];

			// find coordinate system
			vec2f v01(v1 - v0);
			vec2f v01n(v01);
			v01n.normalize();
			vec2f v01_rot90(v01.y(), -v01.x());
			//vec2f v01_rot90n(v01_rot90);
			//v01_rot90n.normalize();

			// express v2 in coordinate system
			vec2f vlocal(v2 - v0);
			float fx = vlocal.dot(v01) / v01.squaredNorm();
			float fy = vlocal.dot(v01_rot90) / v01_rot90.squaredNorm();

			t.tri_coords[j] = vec2f(fx, fy);
		}
	}
}

void Igarashi05::PrecomputeOrientationMatrix()
{
	// Now, let's fill in the matrix G
	// let's explain a little logic here
	// according to Eq(1), let the coordinate of v0, v1 and v2 be (v0x, v0y), (v1x, v1y) and (v2x, v2y), respectively
	// so Eq(1) can be rewrited as
	// (v2x, v2y) = (v0x, v0y) + x * (v1x - v0x, v1y - v0y) + y * (v1y - v0y, v0x - v1x)
	// thus Eq(3) can be rewrited as
	// E = ||((1 - x) * v0x - y * v0y + x * v1x + y * v1y - v2x, y * v0x + (1 - x) * v0y - y * v1x + x * v1y - v2y)||^2
	//   = v' * G * v
	//   = (v0x, v0y, v1x, v1y, v2x, v2y)' * G * (v0x, v0y, v1x, v1y, v2x, v2y)
	// where G = A' * A, and
	// A = [1 - x,    -y,  x, y, -1,  0;
	//          y, 1 - x, -y, x,  0, -1]
	// So G =
	//[ (x - 1)*(x - 1) + y*y,                     0, - x*(x - 1) - y*y,                 y, x - 1,    -y;
	//                      0, (x - 1)*(x - 1) + y*y,                -y, - x*(x - 1) - y*y,     y, x - 1;
	//      - x*(x - 1) - y*y,                    -y,         x*x + y*y,         y*x - x*y,    -x,     y;
	//        			    y,     - x*(x - 1) - y*y,         x*y - y*x,         x*x + y*y,    -y,    -x;
	//                  x - 1,                     y,                -x,                -y,     1,     0;
	//                     -y,                 x - 1,                 y,                -x,     0,     1]
	// and E = v' * G * v
	//       = v' * Gtri * v,
	// where Gtri =
	//[ (x - 1)*(x - 1) + y*y,                     0, -2x*(x - 1) - 2y*y,                 2y,   2x - 2,    -2y;
	//                      0, (x - 1)*(x - 1) + y*y,                -2y, -2x*(x - 1) - 2y*y,       2y, 2x - 2;
	//						0,                     0,          x*x + y*y,                  0,      -2x,     2y;
	//        			    0,                     0,                  0,          x*x + y*y,      -2y,    -2x;
	//                      0,                     0,                  0,                  0,        1,      0;
	//                      0,                     0,                  0,                  0,        0,      1]
	for (auto& t : m_tris)
	{
		for (size_t j = 0; j < 3; ++j)
		{
			int n0x = 2 * m_vertex_map[t.idx_verts[j]];
			int n0y = n0x + 1;
			int n1x = 2 * m_vertex_map[t.idx_verts[(j + 1) % 3]];
			int n1y = n1x + 1;
			int n2x = 2 * m_vertex_map[t.idx_verts[(j + 2) % 3]];
			int n2y = n2x + 1;
			float x = t.tri_coords[j].x();
			float y = t.tri_coords[j].y();

			// n0x,n?? elems, the first line of matrix Gtri as explained earlier
			m_G(n0x, n0x) += 1 - 2 * x + x * x + y * y;
			m_G(n0x, n1x) += 2 * x - 2 * x*x - 2 * y*y;
			m_G(n0x, n1y) += 2 * y;
			m_G(n0x, n2x) += -2 + 2 * x;
			m_G(n0x, n2y) += -2 * y;

			// n0y,n?? elems, the second line of matrix Gtri
			m_G(n0y, n0y) += 1 - 2 * x + x * x + y * y;
			m_G(n0y, n1x) += -2 * y;
			m_G(n0y, n1y) += 2 * x - 2 * x*x - 2 * y*y;
			m_G(n0y, n2x) += 2 * y;
			m_G(n0y, n2y) += -2 + 2 * x;

			// n1x,n?? elems, the third line of matrix Gtri
			m_G(n1x, n1x) += x * x + y * y;
			m_G(n1x, n2x) += -2 * x;
			m_G(n1x, n2y) += 2 * y;

			// n1y,n?? elems, the fourth line of matrix Gtri
			m_G(n1y, n1y) += x * x + y * y;
			m_G(n1y, n2x) += -2 * y;
			m_G(n1y, n2y) += -2 * x;

			// final 2 elems, the fifth and sixth line of Gtri
			m_G(n2x, n2x) += 1;
			m_G(n2y, n2y) += 1;
		}
	}

	size_t n_cons = m_constraints.size();
	size_t n_free = m_verts.size() - n_cons;

	// extract G00, G01 and G10 from G, and then compute Gprime and B
	Eigen::MatrixXd mG00(2 * n_free, 2 * n_free);
	Eigen::MatrixXd mG01(2 * n_free, 2 * n_cons);
	Eigen::MatrixXd mG10(2 * n_cons, 2 * n_free);

	ExtractSubMatrix( m_G, 0, 0, mG00 );
	ExtractSubMatrix( m_G, 0, 2*n_free, mG01 );
	ExtractSubMatrix( m_G, 2*n_free, 0, mG10 );

	// ok, now compute GPrime = G00 + Transpose(G00) and B = G01 + Transpose(G10)
	m_Gprime = mG00 + mG00.transpose();
	m_B = mG01 + mG10.transpose();

	Eigen::MatrixXd m_GprimeInverse = m_Gprime.inverse();
	Eigen::MatrixXd mFinal = m_GprimeInverse * m_B;

	m_first_mat = (-1) * mFinal;
}

void Igarashi05::PrecomputeScalingMatrices(Triangle& t)
{
	// ok now fill tri matrix
	t.F.resize(4, 4);
	t.C.resize(4, 6);

	// precompute coeffs
	double x01 = t.tri_coords[0].x();
	double y01 = t.tri_coords[0].y();
	double x12 = t.tri_coords[1].x();
	double y12 = t.tri_coords[1].y();
	double x20 = t.tri_coords[2].x();
	double y20 = t.tri_coords[2].y();

	double k1 = x12*y01 + (-1 + x01)*y12;
	double k2 = -x12 + x01*x12 - y01*y12;
	double k3 = -y01 + x20*y01 + x01*y20;
	double k4 = -y01 + x01*y01 + x01*y20;
	double k5 = -x01 + x01*x20 - y01*y20 ;

	double a = -1 + x01;
	double a1 = pow(-1 + x01,2) + pow(y01,2);
	double a2 = pow(x01,2) + pow(y01,2);
	double b =  -1 + x20;
	double b1 = pow(-1 + x20,2) + pow(y20,2);
	double c2 = pow(x12,2) + pow(y12,2);

	double r1 = 1 + 2*a*x12 + a1*pow(x12,2) - 2*y01*y12 + a1*pow(y12,2);
	double r2 = -(b*x01) - b1*pow(x01,2) + y01*(-(b1*y01) + y20);
	double r3 = -(a*x12) - a1*pow(x12,2) + y12*(y01 - a1*y12);
	double r5 = a*x01 + pow(y01,2);
	double r6 = -(b*y01) - x01*y20;
	double r7 = 1 + 2*b*x01 + b1*pow(x01,2) + b1*pow(y01,2) - 2*y01*y20;

	//  set up F matrix

	// row 0 mF
	t.F(0, 0) = 2*a1 + 2*a1*c2 + 2*r7;
	t.F(0, 1) = 0;
	t.F(0, 2) = 2*r2 + 2*r3 - 2*r5;
	t.F(0, 3) = 2*k1 + 2*r6 + 2*y01;

	// row 1
	t.F(1, 0) = 0;
	t.F(1, 1) = 2*a1 + 2*a1*c2 + 2*r7;
	t.F(1, 2) = -2*k1 + 2*k3 - 2*y01;
	t.F(1, 3) = 2*r2 + 2*r3 - 2*r5;

	// row 2
	t.F(2, 0) = 2*r2 + 2*r3 - 2*r5;
	t.F(2, 1) = -2*k1 + 2*k3 - 2*y01;
	t.F(2, 2) = 2*a2 + 2*a2*b1 + 2*r1;
	t.F(2, 3) = 0;

	//row 3
	t.F(3, 0) = 2*k1 - 2*k3 + 2*y01;
	t.F(3, 1) = 2*r2 + 2*r3 - 2*r5;
	t.F(3, 2) = 0;
	t.F(3, 3) = 2*a2 + 2*a2*b1 + 2*r1;

	// ok, now invert F
	t.F = t.F.inverse() * -1.0;

	// set up C matrix

	// row 0 mC
	t.C(0, 0) = 2*k2;
	t.C(0, 1) = -2*k1;
	t.C(0, 2) = 2*(-1-k5);
	t.C(0, 3) = 2*k3;
	t.C(0, 4) = 2*a;
	t.C(0, 5) = -2*y01;

	// row 1 mC
	t.C(1, 0) = 2*k1;
	t.C(1, 1) = 2*k2;
	t.C(1, 2) = -2*k3;
	t.C(1, 3) = 2*(-1-k5);
	t.C(1, 4) = 2*y01;
	t.C(1, 5) = 2*a;

	// row 2 mC
	t.C(2, 0) = 2*(-1-k2);
	t.C(2, 1) = 2*k1;
	t.C(2, 2) = 2*k5;
	t.C(2, 3) = 2*r6;
	t.C(2, 4) = -2*x01;
	t.C(2, 5) = 2*y01;

	// row 3 mC
	t.C(3, 0) = 2*k1;
	t.C(3, 1) = 2*(-1-k2);
	t.C(3, 2) = -2*k3;
	t.C(3, 3) = 2*k5;
	t.C(3, 4) = -2*y01;
	t.C(3, 5) = -2*x01;
}

void Igarashi05::PrecomputeFittingMatrices()
{
	auto n_vert = m_deformed.size();

	// make Hy and Hx matrices
	Eigen::MatrixXd HX(n_vert, n_vert);
	Eigen::MatrixXd HY(n_vert, n_vert);
	HX.setZero();
	HY.setZero();

	// ok, now fill matrix
	for (auto& t : m_tris)
	{
		for ( int j = 0; j < 3; ++j )
		{
			int nA = m_vertex_map[t.idx_verts[j]];
			int nB = m_vertex_map[t.idx_verts[(j+1)%3]];

			// X elems
			HX(nA, nA) += 2;
			HX(nA, nB) += -2;
			HX(nB, nA) += -2;
			HX(nB, nB) += 2;

			//  Y elems
			HY(nA, nA) += 2;
			HY(nA, nB) += -2;
			HY(nB, nA) += -2;
			HY(nB, nB) += 2;
		}
	}

	size_t n_cons = m_constraints.size();
	size_t n_free = n_vert - n_cons;

	// extract HX00 and  HY00 matrices
	Eigen::MatrixXd HX00(n_free, n_free);
	Eigen::MatrixXd HY00(n_free, n_free);
	ExtractSubMatrix(HX, 0, 0, HX00);
	ExtractSubMatrix(HY, 0, 0, HY00);

	// Extract HX01 and HX10 matrices
	Eigen::MatrixXd HX01(n_free, n_cons);
	Eigen::MatrixXd HX10(n_cons, n_free);
	ExtractSubMatrix(HX, 0, n_free, HX01);
	ExtractSubMatrix(HX, n_free, 0, HX10);

	// Extract HY01 and HY10 matrices
	Eigen::MatrixXd HY01(n_free, n_cons);
	Eigen::MatrixXd HY10(n_cons, n_free);
	ExtractSubMatrix(HY, 0, n_free, HY01);
	ExtractSubMatrix(HY, n_free, 0, HY10);

	// now compute HXPrime = HX00 + Transpose(HX00) (and HYPrime)
	//Wml::GMatrixd HXPrime( HX00 + HX00.Transpose() );
	//Wml::GMatrixd HYPrime( HY00 + HY00.Transpose() );
	m_HXPrime = HX00;
	m_HYPrime = HY00;

	// and then D = HX01 + Transpose(HX10)
	//Wml::GMatrixd mDX = HX01 + HX10.Transpose();
	//Wml::GMatrixd mDY = HY01 + HY10.Transpose();
	m_DX = HX01;
	m_DY = HY01;

	// pre-compute LU decompositions
	Eigen::FullPivLU<Eigen::MatrixXd> lu(n_free, n_free);
	lu.compute(m_HXPrime);
	m_LUDecompX = lu.matrixLU();
	lu.compute(m_HYPrime);
	m_LUDecompY = lu.matrixLU();
}

void Igarashi05::UpdateScaledTriangle(Triangle& t)
{
	// multiply mC by deformed vertex position
	auto& dv0 = m_deformed[t.idx_verts[0]];
	auto& dv1 = m_deformed[t.idx_verts[1]];
	auto& dv2 = m_deformed[t.idx_verts[2]];
	Eigen::VectorXd deformed(6);
	deformed << dv0.x(), dv0.y(), dv1.x(), dv1.y(), dv2.x(), dv2.y();
	auto c_vec = t.C * deformed;

	// compute -MFInv * mC
	auto solution = t.F * c_vec;

	// ok, grab deformed v0 and v1 from solution vector
	vec2f fitted0((float)solution[0], (float)solution[1]);
	vec2f fitted1((float)solution[2], (float)solution[3]);

	// figure out fitted2
	float x01 = t.tri_coords[0].x();
	float y01 = t.tri_coords[0].y();
	vec2f fitted01(fitted1 - fitted0);
	vec2f fitted01_perp(fitted01.y(), -fitted01.x() );
	vec2f fitted2(fitted0 + (float)x01 * fitted01 + (float)y01 * fitted01_perp);

	// ok now determine scale
	auto& ori_v0 = m_verts[t.idx_verts[0]];
	auto& ori_v1 = m_verts[t.idx_verts[1]];
	float scale = (ori_v1 - ori_v0).norm() / fitted01.norm();

	// now scale triangle
	t.scaled[0] = fitted0 * scale;
	t.scaled[1] = fitted1 * scale;
	t.scaled[2] = fitted2 * scale;
}

void Igarashi05::ApplyFittingStep()
{
	auto n_vert = m_deformed.size();

	// make vector of deformed vertex weights
	Eigen::VectorXd FX(n_vert);
	Eigen::VectorXd FY(n_vert);
	FX.setZero();
	FY.setZero();

	for (auto& t : m_tris)
	{
		for ( int j = 0; j < 3; ++j )
		{
			int A = m_vertex_map[t.idx_verts[j]];
			int B = m_vertex_map[t.idx_verts[(j+1)%3]];

			vec2f def_a(t.scaled[j]);
			vec2f def_b(t.scaled[(j+1)%3]);

			// X elems
			FX[A] += -2 * def_a.x() + 2 * def_b.x();
			FX[B] +=  2 * def_a.x() - 2 * def_b.x();

			//  Y elems
			FY[A] += -2 * def_a.y() + 2 * def_b.y();
			FY[B] +=  2 * def_a.y() - 2 * def_b.y();
		}
	}

	// make F0 vectors
	size_t n_cons = m_constraints.size();
	size_t n_free = n_vert - n_cons;
	Eigen::VectorXd F0X(n_free), F0Y(n_free);
	for (size_t i = 0; i < n_free; ++i) {
		F0X[i] = FX[i];
		F0Y[i] = FY[i];
	}

	// make Q vectors (vectors of constraints)
	Eigen::VectorXd QX(n_cons),  QY(n_cons);
	auto itr = m_constraints.begin();
	for (size_t i = 0; i < n_cons; ++i) {
		QX[i] = m_deformed[*itr].x();
		QY[i] = m_deformed[*itr].y();
		++itr;
	}

	// ok, compute RHS for X and solve
	Eigen::VectorXd RHSX(m_DX * QX);
	RHSX += F0X;
	RHSX *= -1;
	Eigen::VectorXd SolutionX(n_free);
	//Wml::LinearSystemd::Solve( m_mHXPrime, RHSX, SolutionX );
	SolutionX = m_LUDecompX.lu().solve(RHSX);

	// now for Y
	Eigen::VectorXd RHSY(m_DY * QY);
	RHSY += F0Y;
	RHSY *= -1;
	Eigen::VectorXd SolutionY(n_free);
//	Wml::LinearSystemd::Solve( m_mHYPrime, RHSY, SolutionY );
	SolutionY = m_LUDecompY.lu().solve(RHSY);

	// done!
	for (size_t i = 0; i < n_vert; ++i)
	{
		if (m_constraints.find(i) != m_constraints.end()) {
			continue;
		}
		int row = m_vertex_map[i];
		m_deformed[i] = vec2f(SolutionX[row], SolutionY[row]);
	}
}

void Igarashi05::UpdateMesh(bool rigid)
{
	size_t sz = m_constraints.size();
	Eigen::VectorXd q(sz * 2);
	int ptr = 0;
	for (auto& c : m_constraints)
	{
		q[ptr * 2]     = m_deformed[c].x();
		q[ptr * 2 + 1] = m_deformed[c].y();
		++ptr;
	}

	auto u = m_first_mat * q;
	for (size_t i = 0, n = m_verts.size(); i < n; ++i)
	{
		if (m_constraints.find(i) != m_constraints.end()) {
			continue;
		}
		size_t row = m_vertex_map[i];
		double x = u[row * 2];
		double y = u[row * 2 + 1];
		m_deformed[i] = vec2f(x, y);
	}

	if (rigid)
	{
		for (auto& tri : m_tris) {
			PrecomputeScalingMatrices(tri);
		}
		ApplyFittingStep();
	}
}

}