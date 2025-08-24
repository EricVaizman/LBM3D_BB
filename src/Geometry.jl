module Geometry

using ..Parameters
using CUDA

export uw_at_wall_lat_planar, phi_world_lat_3Dplanar, generate_mask_s_local!, generate_mask_bb_local!, compile_bb_data!, generate_plotmask, SolidMaskState, init_solid_mask_state

# In this project, the generated geometry is in fact a circular cylinder with radius R,
# which extends all the way across the Z (width) direction, and is centered at (x_c, y_c)
# In future versions there will be added support for different types of geometries


struct SolidMaskState
    shape_flag::Int8      # 1=cylinder, 2=airfoil
    R2::Float32           # cylinder: R^2; otherwise 0
    chord::Float32        # airfoil chord (phys units); 0 for cylinder
    m::Float32            # NACA camber
    p::Float32            # NACA camber location
    t::Float32            # NACA thickness
    x_int::Float32        # sharp-TE squeeze (e.g., 1.0089304f0)
    x_le_offset::Float32  # pivot→LE distance along +chord (x_ref_frac * chord)
end


function init_solid_mask_state(params::Parameters.LBMParams;
                               x_ref_frac::Float32 = (params.shape === :AIR ? 0.25f0 : 0f0),
                               x_int::Float32 = 1.0089304f0)
    if params.shape === :CYL
        return SolidMaskState(Int8(1), (params.R*params.R), 0f0, 0f0, 0f0, 0f0, x_int, 0f0)
    elseif params.shape === :AIR
        @assert params.chord > 0f0 "Airfoil selected but chord ≤ 0."
        m,p,t = _parse_naca4(something(params.naca_code, "0012"))
        x_le_offset = x_ref_frac * params.chord
        return SolidMaskState(Int8(2), 0f0, params.chord, m, p, t, x_int, x_le_offset)
    else
        error("Unknown shape=$(params.shape). Expected :CYL or :AIR")
    end
end




# ---------------------------------------------------




function generate_mask_s_local!_gpu(
    mask_s_local::CuDeviceVector{Bool},
    x_lli::CuDeviceVector{Float32},
    y_lli::CuDeviceVector{Float32},
    x_c::Float32, y_c::Float32, cθ::Float32, sθ::Float32,
    shape_flag::Int8,
    R2::Float32,
    chord::Float32, m::Float32, p::Float32, t::Float32,
    x_int::Float32,
    x_le_offset::Float32,
    N::Int32
)
    a = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if !(1 <= a <= N)
        return
    end

    dx = x_lli[a] - x_c
    dy = y_lli[a] - y_c

    inside = false

    if shape_flag == Int8(1)
        # Cylinder centered at pivot
        inside = (dx*dx + dy*dy) <= (R2 + 1f-6*R2)
    else
        # Airfoil about pivot
        xB, yB = _rot_to_body(dx, dy, cθ, sθ)
        xB_le  = xB + x_le_offset
        if 0f0 <= xB_le <= chord
            yL, yU = _naca_upper_lower_y(xB_le, chord, m, p, t, x_int)
            tol = 1f-6 * chord
            inside = (yL - tol <= yB <= yU + tol)
        end
    end

    @inbounds mask_s_local[a] = inside
    return
end




function generate_mask_s_local!(
    mask_s_local::CuArray{Bool},
    x_lli::CuArray{Float32},
    y_lli::CuArray{Float32},
    x_c::Float32, y_c::Float32,
    cθ::Float32, sθ::Float32,          # host-computed cos/sin(theta_now)
    state::SolidMaskState
)
    N = Int32(length(mask_s_local))
    @assert N == length(x_lli) == length(y_lli)

    threads = 256
    blocks  = cld(N, threads)
    @cuda threads=threads blocks=blocks generate_mask_s_local!_gpu(
        mask_s_local, x_lli, y_lli,
        x_c, y_c, cθ, sθ,
        state.shape_flag,
        state.R2,
        state.chord, state.m, state.p, state.t,
        state.x_int,
        state.x_le_offset,
        N
    )

    return mask_s_local
end




function generate_mask_bb_local!_gpu(
    mask_bb_local::CuDeviceVector{Bool},
    mask_s_local::CuDeviceVector{Bool},
    nbr_lli::CuDeviceMatrix{Int32},
    q::Int32,
    N_owned::Int32
)
    a = (blockIdx().x-1)*blockDim().x + threadIdx().x   # owned LLI
    if 1 <= a <= N_owned
        if mask_s_local[a]
            mask_bb_local[a] = false           # solid nodes are not BB fluid nodes
            return
        end
        # fluid owned node: BB if any neighbor is solid
        hit = false
        @inbounds for qdir in 1:q
            b = nbr_lli[qdir, a]
            if (b != -1) && (mask_s_local[b])
                hit = true
                break
            end
        end
        mask_bb_local[a] = hit
    end

    return
end




function generate_mask_bb_local!(
    mask_bb_local::CuArray{Bool},          # length N_owned
    mask_s_local::CuArray{Bool},           # length N_owned + N_halo
    nbr_lli::CuArray{Int32,2},     # size (q, N_owned)
)

    q = Int32(size(nbr_lli, 1))
    N_owned = Int32(size(nbr_lli, 2))
    @assert length(mask_bb_local) == N_owned
    threads = 256
    blocks  = cld(N_owned, threads)
    @cuda threads=threads blocks=blocks generate_mask_bb_local!_gpu(
        mask_bb_local,
        mask_s_local,
        nbr_lli,
        q,
        N_owned
    )
        
    return mask_bb_local
end




# --- 1) Count BB links per owned node (O(q*N_owned)) ---
# cnt[a] = number of directions qdir where neighbor b is solid
# (only for fluid owned nodes that are BB; else 0)
function kernel_count_bb_links!(
    cnt::CuDeviceVector{Int32},           # length N_owned
    mask_s::CuDeviceVector{Bool},         # length N_local (owned+halo)
    mask_bb::CuDeviceVector{Bool},        # length N_owned
    nbr_lli::CuDeviceMatrix{Int32},       # (q, N_owned)
    q::Int32, N_owned::Int32
)
    a = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if 1 <= a <= N_owned
        if !mask_bb[a]
            cnt[a] = 0
            return
        end
        c = Int32(0)
        @inbounds for qdir in 1:q
            b = nbr_lli[qdir, a]
            if b != -1 && mask_s[b]
                c += 1
            end
        end
        cnt[a] = c
    end
    return
end




# --- 2) Build dense CSR pointer from inclusive scan (GPU-only) ---
# We first do an inclusive scan on cnt (in-place), then form bb_ptr:
#   bb_ptr[1]      = 1
#   bb_ptr[a+1]    = 1 + cnt[a]           (cnt is now inclusive)
# nnz = cnt[N_owned]
function kernel_build_ptr_from_inclusive!(
    bb_ptr::CuDeviceVector{Int32},        # length N_owned+1
    cnt_scan::CuDeviceVector{Int32},      # length N_owned (inclusive scan of counts)
    N_owned::Int32
)
    a = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if a == 1
        @inbounds bb_ptr[1] = Int32(1)
    end
    if 1 <= a <= N_owned
        @inbounds bb_ptr[a+1] = Int32(1) + cnt_scan[a]
    end
    return
end




# --- 3) Fill CSR (no atomics; each thread writes its own segment) ---
function kernel_fill_bb_csr!(
    bb_dirs::CuDeviceVector{Int8},        # capacity >= nnz
    bb_lidxs::CuDeviceVector{Int32},      # capacity >= nnz
    bb_ptr::CuDeviceVector{Int32},        # length N_owned+1
    mask_s::CuDeviceVector{Bool},         # length N_local
    nbr_lli::CuDeviceMatrix{Int32},       # (q, N_owned)
    q::Int32, N_owned::Int32
)
    a = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if 1 <= a <= N_owned
        start = @inbounds bb_ptr[a]
        stop  = @inbounds bb_ptr[a+1]
        if start == stop
            return
        end
        i = start
        @inbounds for qdir in 1:q
            b = nbr_lli[qdir, a]
            if b != -1 && mask_s[b]
                bb_dirs[i]  = Int8(qdir)
                bb_lidxs[i] = Int32(a)
                i += 1
                # Optional safety (remove when confident):
                if i > stop; return; end
            end
        end
    end
    return
end




# --- Public wrapper: GPU-only end-to-end build ---
# Returns nnz (Int32). Uses GPU inclusive scan via CUDA.cumsum! on cnt.
function compile_bb_data!(
    bb_dirs::CuArray{Int8},               # OUT (capacity >= nnz)
    bb_lidxs::CuArray{Int32},             # OUT (capacity >= nnz)
    bb_ptr::CuArray{Int32},               # OUT length N_owned+1
    cnt::CuArray{Int32},                  # SCRATCH length N_owned
    mask_s::CuArray{Bool},                # IN length N_local
    mask_bb::CuArray{Bool},               # IN length N_owned
    nbr_lli::CuArray{Int32,2}             # IN (q, N_owned)
)::Int32

    q        = Int32(size(nbr_lli, 1))
    N_owned  = Int32(size(nbr_lli, 2))
    @assert length(mask_bb) == N_owned
    @assert length(cnt)      == N_owned
    @assert length(bb_ptr)   == N_owned + 1

    threads_owned = 256
    blocks_owned  = cld(Int(N_owned), threads_owned)

    # 1) count links per owned node
    @cuda threads=threads_owned blocks=blocks_owned kernel_count_bb_links!(cnt, mask_s, mask_bb, nbr_lli, q, N_owned)

    # 2) inclusive scan on GPU
    #    (cnt becomes inclusive counts)
    CUDA.cumsum!(cnt, cnt)   # in-place inclusive scan on device

    # 3) build ptr from inclusive scan on GPU
    @cuda threads=threads_owned blocks=blocks_owned kernel_build_ptr_from_inclusive!(bb_ptr, cnt, N_owned)

    # 4) get nnz = inclusive_sum[N_owned]
    #    (copying a single Int32 to host is trivial; you may also keep nnz in a 1-element CuArray)
    nnz = Int32(Array(cnt)[end])

    # 5) fill CSR (owned nodes write their segment)
    @cuda threads=threads_owned blocks=blocks_owned kernel_fill_bb_csr!(bb_dirs, bb_lidxs, bb_ptr, mask_s, nbr_lli, q, N_owned)

    return bb_lidxs, bb_dirs, bb_ptr, nnz
end








# ---------- HELPERS -----------
@inline function _rot_to_body(dx::Float32, dy::Float32, cθ::Float32, sθ::Float32)
    return (cθ*dx + sθ*dy), (-sθ*dx + cθ*dy)   # rotation by -theta about pivot
end

@inline function _naca_thickness_y(xbar::Float32, c::Float32, t::Float32, x_int::Float32)
    xbar_int = max(0f0, x_int * xbar)
    rt = sqrt(xbar_int)
    return 5f0 * t * c * (0.2969f0*rt - 0.1260f0*xbar_int - 0.3516f0*xbar_int*xbar_int +
                          0.2843f0*xbar_int^3 - 0.1015f0*xbar_int^4)
end

@inline function _naca_camber_y_dydx(xbar::Float32, m::Float32, p::Float32)
    if m == 0f0 || p == 0f0
        return 0f0, 0f0
    elseif xbar < p
        yc   = m/(p*p) * (2f0*p*xbar - xbar*xbar)
        dycb = 2f0*m/(p*p) * (p - xbar)
        return yc, dycb
    else
        den  = (1f0 - p); den2 = den*den
        yc   = m/(den2) * ((1f0 - 2f0*p) + 2f0*p*xbar - xbar*xbar)
        dycb = 2f0*m/(den2) * (p - xbar)
        return yc, dycb
    end
end

@inline function _naca_upper_lower_y(x::Float32, c::Float32, m::Float32, p::Float32, t::Float32, x_int::Float32)
    xbar = x / c
    yt   = _naca_thickness_y(xbar, c, t, x_int)
    yc, dyc_dxb = _naca_camber_y_dydx(xbar, m, p)
    dyc_dx = dyc_dxb / c
    θz = atan(dyc_dx)
    yU = yc + yt*cos(θz)
    yL = yc - yt*cos(θz)
    return yL, yU
end

@inline function _parse_naca4(code::AbstractString)
    @assert length(code) == 4 "NACA code must be 4 digits like \"2412\""
    d1 = Float32(code[1]-'0'); d2 = Float32(code[2]-'0')
    d3 = Float32(code[3]-'0'); d4 = Float32(code[4]-'0')
    m = d1 / 100f0
    p = d2 / 10f0
    t = (10f0*d3 + d4) / 100f0
    return m, p, t
end

@inline function phi_cylinder_lat(x::Float32, y::Float32,
                                  xc::Float32, yc::Float32, R::Float32)
    dx = x - xc;  dy = y - yc
    return sqrt(dx*dx + dy*dy) - R
end

# Airfoil (extruded in z). Quasi-SDF: negative inside [yL,yU], positive outside.
@inline function phi_airfoil_lat(xw::Float32, yw::Float32,
                                 xc::Float32, yc::Float32,
                                 cθ::Float32, sθ::Float32,
                                 chord::Float32, m::Float32, p::Float32, t::Float32,
                                 x_int::Float32, x_le_offset::Float32)
    dx = xw - xc;  dy = yw - yc
    xB, yB = _rot_to_body(dx, dy, cθ, sθ)
    xB_le  = xB + x_le_offset

    if xB_le < 0f0
        return hypot(-xB_le, yB)                     # ahead of LE
    elseif xB_le > chord
        return hypot(xB_le - chord, yB)              # behind TE
    else
        yL, yU = _naca_upper_lower_y(xB_le, chord, m, p, t, x_int)
        if (yB >= yL) & (yB <= yU)
            return -min(yU - yB, yB - yL)            # inside → negative
        else
            return (yB > yU) ? (yB - yU) : (yL - yB) # vertical distance to nearest surface
        end
    end
end

# Shape switch: shape_flag==1 → cylinder; else airfoil. z is ignored (extrusion).
@inline function phi_world_lat_3Dplanar(x::Float32, y::Float32,
                                        shape_flag::Int8,
                                        xc::Float32, yc::Float32,
                                        R::Float32,
                                        chord::Float32, m::Float32, p::Float32, t::Float32,
                                        x_int::Float32, x_le_offset::Float32,
                                        cθ::Float32, sθ::Float32)
    return (shape_flag == Int8(1)) ?
        phi_cylinder_lat(x, y, xc, yc, R) :
        phi_airfoil_lat(x, y, xc, yc, cθ, sθ, chord, m, p, t, x_int, x_le_offset)
end

# Rigid wall velocity at x_w (planar motion): u_w = (U,V,0) + (0,0,ω)×(r_x,r_y,r_z)
@inline function uw_at_wall_lat_planar(xw::Float32, yw::Float32,
                                       xc::Float32, yc::Float32,
                                       Uc_x::Float32, Uc_y::Float32, ω::Float32)
    rx = xw - xc
    ry = yw - yc
    # Ω = (0,0,ω) ⇒ Ω×r = (-ω ry, +ω rx, 0)
    return (Uc_x - ω*ry,  Uc_y + ω*rx)    # (u_wx, u_wy); u_wz ≡ 0
end








function generate_plotmask(params::LBMParams,
                           c_x::Vector{Int8}, c_y::Vector{Int8}, c_z::Vector{Int8})

    Nx, Ny, Nz = params.Nx, params.Ny, params.Nz
    h          = params.h
    R2         = params.R^2
    x_c        = params.x_c
    y_c        = params.y_c

    # Solid mask (entire domain)
    mask_s_tot  = falses(Nx, Ny, Nz)
    @inbounds @views for iy in 1:Ny, ix in 1:Nx
        x = (ix - 0.5) * h
        y = (iy - 0.5) * h
        if (x - x_c)^2 + (y - y_c)^2 < R2
            mask_s_tot[ix, iy, :] .= true
        end
    end

    # Bounce-back-neighbor mask (fluid nodes adjacent to a solid)
    q            = length(c_x)
    mask_bb_tot  = falses(Nx, Ny, Nz)
    @inbounds for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
        if !mask_s_tot[ix, iy, iz]  # only consider fluid nodes
            for qdir in 1:q
                jx = ix + c_x[qdir]
                jy = iy + c_y[qdir]
                jz = iz + c_z[qdir]
                if 1 <= jx <= Nx && 1 <= jy <= Ny && 1 <= jz <= Nz
                    if mask_s_tot[jx, jy, jz]
                        mask_bb_tot[ix, iy, iz] = true
                        break
                    end
                end
            end
        end
    end

    # If Z is treated as effectively infinite (e.g., Nz>1 with periodic/ghost),
    # copy the neighbor flags from interior slices to the boundaries.
    if Nz != 1
        mask_bb_tot[:, :, 1]  .= mask_bb_tot[:, :, 2]
        mask_bb_tot[:, :, Nz] .= mask_bb_tot[:, :, Nz-1]
    end

    return mask_s_tot, mask_bb_tot
end




end # module