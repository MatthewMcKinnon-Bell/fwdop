from scipy.sparse import csr_matrix, vstack
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.hilbert_space import EuclideanSpace


class GFwdOp(LinearOperator):
    """
    A forward operator that wraps the stacked sensitivity kernel matrix as a
    pygeoinf `LinearOperator`.

    This class collects per-ray sensitivity kernels, stacks them into a single
    sparse matrix `K` (shape: n_observations x n_voxels) and provides the
    linear mapping and its adjoint via the usual matrix-vector products.
    It also preserves per-phase bookkeeping for diagnostics (`getNnz`,
    `get_voxelNum`) and a convenience per-phase apply method.
    """

    def __init__(self, model, rays):
        # store inputs
        self.__model = model
        self.__rays = list(rays)  # preserve input order

        # prepare dict to collect kernels per phase and a list preserving ray order
        self.__kernel_matrices = {}
        self._rows_in_order = []
        self._phases_in_order = []

        # compute kernels and construct stacked sparse matrix
        self.__calcMatrix__()

        # domain: voxels, codomain: observations (stacked rows)
        n_voxels = self._K.shape[1]
        n_obs = self._K.shape[0]
        domain = EuclideanSpace(n_voxels)
        codomain = EuclideanSpace(n_obs)

        # define mapping and adjoint using the stored sparse matrix
        def mapping(x):
            cx = domain.to_components(x)
            cy = self._K.dot(cx)
            return codomain.from_components(cy)

        def adjoint_mapping(y):
            cy = codomain.to_components(y)
            cx = self._K.T.dot(cy)
            return domain.from_components(cx)

        super().__init__(domain, codomain, mapping, adjoint_mapping=adjoint_mapping)

    def __calcMatrix__(self):
        # calculate kernels in the order of rays and track them per phase
        for ray in self.__rays:
            kernel = self.__model.mesh.compute_sensitivity_kernel(
                ray,
                property_name=f"v{ray.phase.name[0].lower()}",
                attach_name=f"K_{ray.phase.name}_v{ray.phase.name[0].lower()}",
                epsilon=1e-6,
            )

            phase = ray.phase.name
            self.__kernel_matrices.setdefault(phase, []).append(kernel)
            self._rows_in_order.append(kernel)
            self._phases_in_order.append(phase)

        # stack rows into one big sparse matrix (observations x voxels)
        if len(self._rows_in_order) > 0:
            self._K = vstack([csr_matrix(r) for r in self._rows_in_order]).tocsr()
        else:
            # empty operator
            self._K = csr_matrix((0, 0))

        # build phase -> slice mapping for diagnostics and per-phase extraction
        self._phase_slices = {}
        start = 0
        for phase in sorted(self.__kernel_matrices.keys()):
            count = len(self.__kernel_matrices[phase])
            self._phase_slices[phase] = slice(start, start + count)
            start += count

        # debug info
        for phase, sl in self._phase_slices.items():
            mat = self._K[sl, :]
            print(f"{phase} Kernel Matrix shape: {mat.shape}, nnz: {mat.nnz}")

    def getNnz(self, phase=["all"]):
        # return number of non-zero entries in kernel matrix
        if phase == ["all"]:
            return {ph: self._K[self._phase_slices[ph], :].nnz for ph in self._phase_slices}
        else:
            return {ph: self._K[self._phase_slices[ph], :].nnz for ph in phase if ph in self._phase_slices}

    def get_voxelNum(self, phase=["all"]):
        # return number of voxels in kernel matrix (columns)
        ncols = self._K.shape[1]
        if phase == ["all"]:
            return {ph: ncols for ph in self._phase_slices}
        else:
            return {ph: ncols for ph in phase if ph in self._phase_slices}

    def __apply__(self, velocity_model):
        # convenience: return per-phase travel times using the stacked matrix
        print("Computing travel times from kernels...")
        times = {}
        for phase, sl in self._phase_slices.items():
            mat = self._K[sl, :]
            if mat.shape[0] > 0:
                vec = mat.dot(velocity_model)
                times[phase] = vec
                print(f"{phase} travel times: min {vec.min():.2f} s, max {vec.max():.2f} s")
        return times
