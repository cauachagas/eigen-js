import createUmfpackModule from './test.mjs';

let umfpackPromise;

function resolveLocateFile(path) {
    if (typeof window !== 'undefined' && typeof window.document !== 'undefined') {
        return new URL(path, import.meta.url).href;
    }
    try {
        const fileUrl = new URL(path, import.meta.url);
        return fileUrl.pathname;
    } catch {
        return path;
    }
}

function validateLengths(nRow, nCol, ap, ai, ax, az, bx, bz) {
    if (nRow !== nCol) {
        throw new Error('This minimal wrapper currently expects a square matrix.');
    }

    if (ap.length !== nCol + 1) {
        throw new Error('Ap must have length nCol + 1.');
    }

    const nnz = ap[ap.length - 1];
    if (ai.length !== nnz || ax.length !== nnz || az.length !== nnz) {
        throw new Error('Ai, Ax, and Az must all have length Ap[nCol].');
    }

    if (bx.length !== nRow || bz.length !== nRow) {
        throw new Error('Bx and Bz must both have length nRow.');
    }
}

function createApi(module) {
    function getHeap32() {
        const heap = module.HEAP32;
        if (!heap || heap.buffer.byteLength === 0) {
            return new Int32Array(module.wasmMemory.buffer);
        }
        return heap;
    }

    function getHeapF64() {
        const heap = module.HEAPF64;
        if (!heap || heap.buffer.byteLength === 0) {
            return new Float64Array(module.wasmMemory.buffer);
        }
        return heap;
    }

    function allocInt32(values) {
        const ptr = module._malloc(values.length * Int32Array.BYTES_PER_ELEMENT);
        if (!ptr) {
            throw new Error('Failed to allocate Int32 buffer in the WebAssembly heap.');
        }

        getHeap32().set(values, ptr >> 2);
        return {
            ptr,
            length: values.length,
            free() {
                module._free(ptr);
            }
        };
    }

    function allocFloat64(values) {
        const ptr = module._malloc(values.length * Float64Array.BYTES_PER_ELEMENT);
        if (!ptr) {
            throw new Error('Failed to allocate Float64 buffer in the WebAssembly heap.');
        }

        getHeapF64().set(values, ptr >> 3);
        return {
            ptr,
            length: values.length,
            free() {
                module._free(ptr);
            }
        };
    }

    function allocZeroFloat64(length) {
        return allocFloat64(new Float64Array(length));
    }

    function readFloat64Array(ptr, length) {
        const heap = getHeapF64();
        const start = ptr >> 3;
        return Array.from(heap.subarray(start, start + length));
    }

    function getInfo(index) {
        return module._wasm_umfpack_get_info(index);
    }

    function callTest(testName) {
        const fn = module[`_test_${testName}`];
        if (typeof fn !== 'function') {
            throw new Error(`Function _test_${testName} not found.`);
        }
        return fn();
    }

    function validateMatrixLengths(nRow, nCol, ap, ai, ax, az) {
        if (nRow !== nCol) {
            throw new Error('This minimal wrapper currently expects a square matrix.');
        }

        if (ap.length !== nCol + 1) {
            throw new Error('Ap must have length nCol + 1.');
        }

        const nnz = ap[ap.length - 1];
        if (ai.length !== nnz || ax.length !== nnz || az.length !== nnz) {
            throw new Error('Ai, Ax, and Az must all have length Ap[nCol].');
        }
    }

    function factorizeComplexSystem({ nRow, nCol, ap, ai, ax, az }) {
        validateMatrixLengths(nRow, nCol, ap, ai, ax, az);

        module._wasm_umfpack_zi_reset();

        const apAlloc = allocInt32(ap instanceof Int32Array ? ap : Int32Array.from(ap));
        const aiAlloc = allocInt32(ai instanceof Int32Array ? ai : Int32Array.from(ai));
        const axAlloc = allocFloat64(ax instanceof Float64Array ? ax : Float64Array.from(ax));
        const azAlloc = allocFloat64(az instanceof Float64Array ? az : Float64Array.from(az));

        const symbolicStatus = module._wasm_umfpack_zi_symbolic(
            nRow,
            nCol,
            apAlloc.ptr,
            aiAlloc.ptr,
            axAlloc.ptr,
            azAlloc.ptr
        );
        if (symbolicStatus < 0) {
            module._wasm_umfpack_zi_reset();
            azAlloc.free();
            axAlloc.free();
            aiAlloc.free();
            apAlloc.free();
            throw new Error(`umfpack_zi_symbolic failed with status ${symbolicStatus}.`);
        }

        const numericStatus = module._wasm_umfpack_zi_numeric(
            apAlloc.ptr,
            aiAlloc.ptr,
            axAlloc.ptr,
            azAlloc.ptr
        );
        if (numericStatus < 0) {
            module._wasm_umfpack_zi_reset();
            azAlloc.free();
            axAlloc.free();
            aiAlloc.free();
            apAlloc.free();
            throw new Error(`umfpack_zi_numeric failed with status ${numericStatus}.`);
        }

        let isFreed = false;

        function solve({ bx, bz, sys = 0 }) {
            if (isFreed) {
                throw new Error('Factorization has already been freed.');
            }
            if (bx.length !== nRow || bz.length !== nRow) {
                throw new Error(`RHS vectors bx and bz must have length ${nRow}.`);
            }

            const bxAlloc = allocFloat64(bx instanceof Float64Array ? bx : Float64Array.from(bx));
            const bzAlloc = allocFloat64(bz instanceof Float64Array ? bz : Float64Array.from(bz));
            const xAlloc = allocZeroFloat64(nCol);
            const xzAlloc = allocZeroFloat64(nCol);

            try {
                const solveStatus = module._wasm_umfpack_zi_solve(
                    sys,
                    apAlloc.ptr,
                    aiAlloc.ptr,
                    axAlloc.ptr,
                    azAlloc.ptr,
                    xAlloc.ptr,
                    xzAlloc.ptr,
                    bxAlloc.ptr,
                    bzAlloc.ptr
                );
                if (solveStatus < 0) {
                    throw new Error(`umfpack_zi_solve failed with status ${solveStatus}.`);
                }

                return {
                    x: readFloat64Array(xAlloc.ptr, nCol),
                    xz: readFloat64Array(xzAlloc.ptr, nCol),
                    status: solveStatus,
                    solveFlops: getInfo(84)
                };
            } finally {
                xzAlloc.free();
                xAlloc.free();
                bzAlloc.free();
                bxAlloc.free();
            }
        }

        function free() {
            if (!isFreed) {
                isFreed = true;
                module._wasm_umfpack_zi_reset();
                azAlloc.free();
                axAlloc.free();
                aiAlloc.free();
                apAlloc.free();
            }
        }

        return {
            nRow,
            nCol,
            nnz: ap[nCol],
            info: {
                symbolicStatus,
                numericStatus,
                rcond: getInfo(67),
                flops: getInfo(84)
            },
            solve,
            free
        };
    }

    function solveComplexSystem(options) {
        validateLengths(
            options.nRow,
            options.nCol,
            options.ap,
            options.ai,
            options.ax,
            options.az,
            options.bx,
            options.bz
        );

        const factor = factorizeComplexSystem(options);
        try {
            const sol = factor.solve(options);
            return {
                x: sol.x,
                xz: sol.xz,
                statuses: {
                    symbolic: factor.info.symbolicStatus,
                    numeric: factor.info.numericStatus,
                    solve: sol.status
                },
                info: {
                    status: getInfo(0),
                    rcond: factor.info.rcond,
                    solveFlops: sol.solveFlops
                }
            };
        } finally {
            factor.free();
        }
    }

    return {
        module,
        callTest,
        getInfo,
        factorizeComplexSystem,
        solveComplexSystem,
        reset() {
            module._wasm_umfpack_zi_reset();
        }
    };
}

export async function loadUmfpack(options = {}) {
    if (!umfpackPromise) {
        umfpackPromise = createUmfpackModule({
            locateFile: resolveLocateFile,
            print: options.print,
            printErr: options.printErr,
            onAbort: options.onAbort
        }).then((module) => createApi(module));
    }

    return umfpackPromise;
}