
#include "udf.h"
#include "math.h"

/* ---------------------------------------------------------
   UDM Layout
   0  N
   1  Σv
   2  Σv²
   3  Σv³
   4  Σsin
   5  Σsin²
   6  Σsin³
   7  Σsin cos
   8  Σcos²
   9  Σα
   10 Σα²
   11 Σv² sin
   12 Σv² sin²
   13 Σv² sin³
   14 Σv² sin cos
   15 Σv² cos²
   16 Σv² α
   17 Σv² α²
   18 Σv^2.5
   19 Σv^2.5 sin
   20 Σv^2.5 sin²
   21 Σv^2.5 sin³
   22 Σv^2.5 sin cos
   23 Σv^2.5 α
   24 Σv^2.5 α²
   25 Σv³ sin
   26 Σv³ sin²
   27 Σv³ sin³
   28 Σv³ sin cos
   29 Σv³ α
   30 Σv³ α²
   31 ΣOka
   32 ΔP_x
   33 ΔP_y
   34 ΔP_z (3D only)
   35 Σ|ΔP|
   36 ΣΔKE
--------------------------------------------------------- */

DEFINE_DPM_BC(custom_rebound, p, t, f, f_normal, dim)
{
   cell_t c = F_C0(f, t);
   Thread *tc = THREAD_T0(t);

   int i;
   real n_mag = NV_MAG(f_normal);
   real n_vec[ND_ND];
   real V_old[ND_ND];

   if (n_mag < 1e-12)
      return PATH_ABORT;

   for (i = 0; i < dim; i++)
   {
      n_vec[i] = f_normal[i] / n_mag;
      V_old[i] = p->state.V[i];
   }

   real v_normal = NV_DOT(p->state.V, n_vec);
   real v_mag = NV_MAG(p->state.V);

   if (v_mag < 1e-12)
      return PATH_ABORT;

   real theta = asin(MAX(0.0, MIN(1.0, fabs(v_normal) / v_mag)));

   real s = sin(theta);
   real cth = cos(theta);
   real s2 = s * s;
   real s3 = s2 * s;
   real c2 = cth * cth;
   real sc = s * cth;

   real v2 = v_mag * v_mag;
   real v25 = pow(v_mag, 2.5);
   real v3 = v2 * v_mag;

   real erosion_func = pow(v_mag, 2.35) *
                       pow(s, 0.8) *
                       pow(1.0 + 1.8 * s, 1.3);

   real en = 0.98913 - 2.0801 * theta + 1.9351 * theta * theta - 0.51128 * theta * theta * theta;

   real et = 1.0104 - 1.4035 * theta + 1.5975 * theta * theta - 0.44414 * theta * theta * theta;

   en = MAX(0.0, MIN(1.0, en));
   et = MAX(0.0, MIN(1.0, et));

   for (i = 0; i < dim; i++)
   {
      real vn = v_normal * n_vec[i];
      real vt = p->state.V[i] - vn;
      p->state.V[i] = -en * vn + et * vt;
   }

   /* Statistical accumulators */
   C_UDMI(c, tc, 0) += 1.0;
   C_UDMI(c, tc, 1) += v_mag;
   C_UDMI(c, tc, 2) += v2;
   C_UDMI(c, tc, 3) += v3;
   C_UDMI(c, tc, 4) += s;
   C_UDMI(c, tc, 5) += s2;
   C_UDMI(c, tc, 6) += s3;
   C_UDMI(c, tc, 7) += sc;
   C_UDMI(c, tc, 8) += c2;
   C_UDMI(c, tc, 9) += theta;
   C_UDMI(c, tc, 10) += theta * theta;

   C_UDMI(c, tc, 11) += v2 * s;
   C_UDMI(c, tc, 12) += v2 * s2;
   C_UDMI(c, tc, 13) += v2 * s3;
   C_UDMI(c, tc, 14) += v2 * sc;
   C_UDMI(c, tc, 15) += v2 * c2;
   C_UDMI(c, tc, 16) += v2 * theta;
   C_UDMI(c, tc, 17) += v2 * theta * theta;

   C_UDMI(c, tc, 18) += v25;
   C_UDMI(c, tc, 19) += v25 * s;
   C_UDMI(c, tc, 20) += v25 * s2;
   C_UDMI(c, tc, 21) += v25 * s3;
   C_UDMI(c, tc, 22) += v25 * sc;
   C_UDMI(c, tc, 23) += v25 * theta;
   C_UDMI(c, tc, 24) += v25 * theta * theta;

   C_UDMI(c, tc, 25) += v3 * s;
   C_UDMI(c, tc, 26) += v3 * s2;
   C_UDMI(c, tc, 27) += v3 * s3;
   C_UDMI(c, tc, 28) += v3 * sc;
   C_UDMI(c, tc, 29) += v3 * theta;
   C_UDMI(c, tc, 30) += v3 * theta * theta;

   C_UDMI(c, tc, 31) += erosion_func;

   /* Momentum transfer */
   real dp[ND_ND], dpmag = 0.0;
   for (i = 0; i < dim; i++)
   {
      dp[i] = P_MASS(p) * (V_old[i] - p->state.V[i]);
      dpmag += dp[i] * dp[i];
   }
   dpmag = sqrt(dpmag);

   C_UDMI(c, tc, 32) += dp[0];
   C_UDMI(c, tc, 33) += dp[1];
#if RP_3D
   C_UDMI(c, tc, 34) += dp[2];
#endif
   C_UDMI(c, tc, 35) += dpmag;

   /* Kinetic energy absorbed */
   {
      real ke0 = 0.0, ke1 = 0.0;
      for (i = 0; i < dim; i++)
      {
         ke0 += V_old[i] * V_old[i];
         ke1 += p->state.V[i] * p->state.V[i];
      }
      ke0 *= 0.5 * P_MASS(p);
      ke1 *= 0.5 * P_MASS(p);

      C_UDMI(c, tc, 36) += (ke0 - ke1);
   }

   return PATH_ACTIVE;
}

DEFINE_INIT(reset_udms, domain)
{
   Thread *t;
   cell_t c;

   thread_loop_c(t, domain){
       begin_c_loop(c, t){
           int i;
   for (i = 0; i <= 36; i++)
      C_UDMI(c, t, i) = 0.0;
}
end_c_loop(c, t)
}

Message("\\n37 UDMs reset successfully.\\n");
}
