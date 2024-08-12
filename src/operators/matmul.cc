#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        const auto &shapeA = inputs[0]->getDims();
        const auto &shapeB = inputs[1]->getDims();
        Shape result;

        int i = 0;
        size_t size = shapeA.size();
        auto fn = [shapeA, shapeB, &result](int i)
        {
            if (shapeA[i] == 1)
            {
                result.emplace_back(shapeB[i]);
            }
            else
            {
                result.emplace_back(shapeA[i]);
            }
        };
        switch (size)
        {
        case 4:
            fn(i++);
        case 3:
            fn(i++);
        default:
            if (transA)
            {
                m = shapeA[i + 1];
                k = shapeA[i];
            }
            else
            {
                m = shapeA[i];
                k = shapeA[i + 1];
            }
            if (transB)
            {
                n = shapeB[i];
            }
            else
            {
                n = shapeB[i + 1];
            }
        }

        result.emplace_back(m);
        result.emplace_back(n);
        return optional<vector<Shape>>({result});
    }

} // namespace infini
