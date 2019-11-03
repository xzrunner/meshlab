#include "meshlab/Levy02.h"

#define GEO_STATIC_LIBS
#include <OpenNL_psm.h>

#include <iostream>

namespace meshlab
{

Levy02::Levy02(const std::vector<vec3d>& verts, const std::vector<size_t>& tris)
    : m_verts(verts)
    , m_tris(tris)
{
}

void Levy02::Apply()
{
    nlNewContext();

    NLuint nb_vertices = NLuint(m_verts.size());

    Project();

    nlSolverParameteri(NL_NB_VARIABLES, NLint(2 * nb_vertices));
    nlSolverParameteri(NL_LEAST_SQUARES, NL_TRUE);
    nlSolverParameteri(NL_MAX_ITERATIONS, NLint(5 * nb_vertices));
    nlSolverParameterd(NL_THRESHOLD, 1e-6);

    nlBegin(NL_SYSTEM);
    MeshToSolver();
    nlBegin(NL_MATRIX);
    SetupLSCM();
    nlEnd(NL_MATRIX);
    nlEnd(NL_SYSTEM);
    std::cout << "Solving ..." << std::endl;

    nlSolve();

    SolverToMesh();
    NormalizeUV();

    double time;
    NLint iterations;
    nlGetDoublev(NL_ELAPSED_TIME, &time);
    nlGetIntegerv(NL_USED_ITERATIONS, &iterations);
    std::cout << "Solver time: " << time << std::endl;
    std::cout << "Used iterations: " << iterations << std::endl;

    nlDeleteContext(nlGetCurrent());
}

void Levy02::Project()
{
    // Get bbox
    double xmin =  1e30;
    double ymin =  1e30;
    double zmin =  1e30;
    double xmax = -1e30;
    double ymax = -1e30;
    double zmax = -1e30;

    for (size_t i = 0; i < m_verts.size(); i++)
    {
        auto& v = m_verts[i];
        xmin = std::min(v.x(), xmin);
        ymin = std::min(v.y(), ymin);
        zmin = std::min(v.z(), zmin);

        xmax = std::max(v.x(), xmax);
        ymax = std::max(v.y(), ymax);
        zmax = std::max(v.z(), zmax);
    }

    double dx = xmax - xmin;
    double dy = ymax - ymin;
    double dz = zmax - zmin;

    vec3d V1,V2;

    // Find shortest bbox axis
    if(dx <= dy && dx <= dz) {
        if(dy > dz) {
            V1 = vec3d(0,1,0);
            V2 = vec3d(0,0,1);
        } else {
            V2 = vec3d(0,1,0);
            V1 = vec3d(0,0,1);
        }
    } else if(dy <= dx && dy <= dz) {
        if(dx > dz) {
            V1 = vec3d(1,0,0);
            V2 = vec3d(0,0,1);
        } else {
            V2 = vec3d(1,0,0);
            V1 = vec3d(0,0,1);
        }
    } else if(dz <= dx && dz <= dy) {
        if(dx > dy) {
            V1 = vec3d(1,0,0);
            V2 = vec3d(0,1,0);
        } else {
            V2 = vec3d(1,0,0);
            V1 = vec3d(0,1,0);
        }
    }

    // Project onto shortest bbox axis,
    // and lock extrema vertices

    double umin = 1e30;
    double umax = -1e30;

    m_uvs.reserve(m_verts.size());
    for (size_t i = 0; i < m_verts.size(); i++)
    {
        auto& V = m_verts[i];
        double u = V.dot(V1);
        double v = V.dot(V2);
        m_uvs.push_back({ u, v });
        if(u < umin) {
            m_vxmin = i;
            umin = u;
        }
        if(u > umax) {
            m_vxmax = i;
            umax = u;
        }
    }
}

void Levy02::ProjectTriangle(const vec3d& p0, const vec3d& p1, const vec3d& p2,
                             vec2d& z0, vec2d& z1, vec2d& z2)
{
    vec3d X = p1 - p0;
    X.normalize();
    vec3d Z = X.cross(p2 - p0);
    Z.normalize();
    vec3d Y = Z.cross(X);
    const vec3d& O = p0;

    double x0 = 0;
    double y0 = 0;
    double x1 = (p1 - O).norm();
    double y1 = 0;
    double x2 = (p2 - O).dot(X);
    double y2 = (p2 - O).dot(Y);

    z0 = vec2d(x0,y0);
    z1 = vec2d(x1,y1);
    z2 = vec2d(x2,y2);
}

void Levy02::MeshToSolver()
{
    assert(m_vxmin >= 0 && m_vxmax >= 0);
    for(NLuint i = 0; i < m_verts.size(); ++i)
    {
        auto& it = m_verts[i];
        double u = m_uvs[i].x();
        double v = m_uvs[i].y();
        nlSetVariable(2 * i    , u);
        nlSetVariable(2 * i + 1, v);
        if (i == m_vxmin || i == m_vxmax) {
            nlLockVariable(2 * i    );
            nlLockVariable(2 * i + 1);
        }
    }
}

void Levy02::SolverToMesh()
{
    for(NLuint i = 0; i < m_verts.size(); ++i)
    {
        auto& it = m_verts[i];
        double u = nlGetVariable(2 * i);
        double v = nlGetVariable(2 * i + 1);
        m_uvs[i] = vec2d(u, v);
    }
}

void Levy02::SetupLSCM()
{
    assert(m_tris.size() % 3 == 0);
    for (size_t i = 0, n = m_tris.size(); i < n; ) {
        auto v0 = m_tris[i++];
        auto v1 = m_tris[i++];
        auto v2 = m_tris[i++];
        SetupConformalMapRelations(v0, v1, v2);
    }
}

void Levy02::SetupConformalMapRelations(size_t v0, size_t v1, size_t v2)
{
    auto& p0 = m_verts[v0];
    auto& p1 = m_verts[v1];
    auto& p2 = m_verts[v2];

    vec2d z0,z1,z2;
    ProjectTriangle(p0,p1,p2,z0,z1,z2);
    vec2d z01 = z1 - z0;
    vec2d z02 = z2 - z0;
    double a = z01.x();
    double b = z01.y();
    double c = z02.x();
    double d = z02.y();
    assert(b == 0.0);

    // Note  : 2*id + 0 --> u
    //         2*id + 1 --> v
    NLuint u0_id = 2*v0    ;
    NLuint v0_id = 2*v0 + 1;
    NLuint u1_id = 2*v1    ;
    NLuint v1_id = 2*v1 + 1;
    NLuint u2_id = 2*v2    ;
    NLuint v2_id = 2*v2 + 1;

    // Note : b = 0

    // Real part
    nlBegin(NL_ROW);
    nlCoefficient(u0_id, -a+c) ;
    nlCoefficient(v0_id,  b-d) ;
    nlCoefficient(u1_id,   -c) ;
    nlCoefficient(v1_id,    d) ;
    nlCoefficient(u2_id,    a);
    nlEnd(NL_ROW);

    // Imaginary part
    nlBegin(NL_ROW);
    nlCoefficient(u0_id, -b+d);
    nlCoefficient(v0_id, -a+c);
    nlCoefficient(u1_id,   -d);
    nlCoefficient(v1_id,   -c);
    nlCoefficient(v2_id,    a);
    nlEnd(NL_ROW);
}

void Levy02::NormalizeUV()
{
    double u_min=1e30, v_min=1e30, u_max=-1e30, v_max=-1e30;
    for(NLuint i=0; i<m_verts.size(); ++i)
    {
	    u_min = std::min(u_min, m_uvs[i].x());
	    v_min = std::min(v_min, m_uvs[i].y());
	    u_max = std::max(u_max, m_uvs[i].x());
	    v_max = std::max(v_max, m_uvs[i].y());
    }

    double l = std::max(u_max-u_min,v_max-v_min);
    for(NLuint i=0; i<m_verts.size(); ++i)
    {
        double u = m_uvs[i].x();
        double v = m_uvs[i].y();
	    u -= u_min;
	    u /= l;
	    v -= v_min;
	    v /= l;
        m_uvs[i] = vec2d(u, v);
    }
}

}