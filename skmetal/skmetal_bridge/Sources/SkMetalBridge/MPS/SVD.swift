import Accelerate

final class MPSSVD {
    /// Compute SVD using Accelerate's LAPACK (sgesdd_).
    /// Input/output are row-major (numpy C order).
    static func compute(
        A: UnsafeMutablePointer<Float>,
        m: Int,
        n: Int,
        k: Int,
        U: UnsafeMutablePointer<Float>,
        S: UnsafeMutablePointer<Float>,
        Vt: UnsafeMutablePointer<Float>
    ) -> Bool {
        var jobz: CChar = 83 // 'S' - economy size
        var m_ = Int32(m)
        var n_ = Int32(n)
        var lda = Int32(max(1, m))
        var ldu = Int32(max(1, m))
        var work: Float = 0.0
        var lwork: Int32 = -1
        var info: Int32 = 0
        let minMN = min(m, n)
        var iwork = [Int32](repeating: 0, count: 8 * minMN)

        // Convert row-major input to column-major for LAPACK
        var aColMajor = [Float](repeating: 0, count: m * n)
        for i in 0..<m {
            for j in 0..<n {
                aColMajor[j * m + i] = A[i * n + j]
            }
        }

        var s = [Float](repeating: 0, count: minMN)
        var uColMajor = [Float](repeating: 0, count: m * minMN)
        var ldvt = Int32(minMN)
        var vtColMajor = [Float](repeating: 0, count: minMN * n)

        // Query optimal workspace
        sgesdd_(&jobz, &m_, &n_, &aColMajor, &lda, &s,
                &uColMajor, &ldu, &vtColMajor, &ldvt,
                &work, &lwork, &iwork, &info)

        lwork = Int32(work)
        var workArray = [Float](repeating: 0, count: Int(lwork))

        // Compute SVD
        sgesdd_(&jobz, &m_, &n_, &aColMajor, &lda, &s,
                &uColMajor, &ldu, &vtColMajor, &ldvt,
                &workArray, &lwork, &iwork, &info)

        guard info == 0 else { return false }

        // Copy top-k singular values
        for i in 0..<k {
            S[i] = s[i]
        }

        // Convert U from column-major (m×minMN) to row-major, take first k columns
        for i in 0..<m {
            for j in 0..<k {
                U[i * k + j] = uColMajor[i + j * m]
            }
        }

        // Convert Vt from column-major (minMN×n) to row-major, take top k rows
        for i in 0..<k {
            for j in 0..<n {
                Vt[i * n + j] = vtColMajor[i + j * minMN]
            }
        }

        return true
    }
}
