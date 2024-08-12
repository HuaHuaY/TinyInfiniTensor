#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include "operators/matmul.h"
#include "operators/transpose.h"

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================

        size_t size;
        size_t size_new;
        std::unordered_set<Operator> del_ops;

        auto optimize_two_transpose = [this, &del_ops](Operator &op)
        {
            if (del_ops.find(op) == del_ops.end() && op->getOpType() == OpType::Transpose)
            {
                auto op1 = static_cast<TransposeObj *>(op.get());
                if (auto targets = op->getOutput(0)->getTargets();
                    targets.size() == 1 &&
                    del_ops.find(targets[0]) == del_ops.end() &&
                    targets[0]->getOpType() == OpType::Transpose)
                {
                    auto op2 = static_cast<TransposeObj *>(targets[0].get());
                    auto vec1 = op1->getPermute();
                    auto vec2 = op2->getPermute();
                    for (size_t i = 0; i < vec1.size(); i++)
                    {
                        if ((int)i != vec2[vec1[i]])
                        {
                            return;
                        }
                    }
                    auto input = op1->inputs[0];
                    auto source = input->getSource();
                    input->removeTarget(op);
                    if (source != nullptr)
                    {
                        source->removeSuccessors(op);
                    }
                    for (auto t : op2->outputs[0]->getTargets())
                    {
                        for (size_t j = 0; j < t->inputs.size(); j++)
                        {
                            if (t->inputs[j] == op2->outputs[0])
                            {
                                t->inputs[j] = input;
                            }
                        }
                        t->removePredecessors(targets[0]);
                        input->addTarget(t);
                        if (source != nullptr)
                        {
                            t->addPredecessors(source);
                            source->addSuccessors(t);
                        }
                    }
                    this->removeTensor(op1->getOutput(0));
                    this->removeTensor(op2->getOutput(0));
                    del_ops.insert(op);
                    del_ops.insert(targets[0]);
                }
            }
        };

        do
        {
            size = this->ops.size();
            for (auto &op : ops)
            {
                optimize_two_transpose(op);
            }
            for (auto &op : del_ops)
            {
                this->removeOperator(op);
            }
            size_new = this->ops.size();
        } while (size != size_new);

        auto optimize_transpose_matmul = [this, &del_ops](Operator &op)
        {
            if (del_ops.find(op) == del_ops.end() && op->getOpType() == OpType::Transpose)
            {
                auto op1 = static_cast<TransposeObj *>(op.get());
                auto permute = op1->getPermute();
                size_t i = 0;
                for (; i < permute.size() - 2; i++)
                {
                    if (permute[i] != (int)i)
                    {
                        return;
                    }
                }
                if (permute[i] != (int)i + 1 && permute[i + 1] != (int)i)
                {
                    return;
                }
                if (auto targets = op->outputs[0]->getTargets();
                    targets.size() == 1 && targets[0]->getOpType() == OpType::MatMul)
                {
                    auto op2 = static_cast<MatmulObj *>(targets[0].get());
                    auto input = op1->getInputs(0);
                    auto source = input->getSource();
                    if (op2->inputs[0] == op1->inputs[0])
                    {
                        op2->setTransA(!op2->getTransA());
                        op2->inputs[0] = input;
                    }
                    else
                    {
                        op2->setTransB(!op2->getTransB());
                        op2->inputs[1] = input;
                    }
                    input->removeTarget(op);
                    input->addTarget(targets[0]);
                    targets[0]->removePredecessors(op);
                    if (source != nullptr)
                    {
                        source->removeSuccessors(op);
                        source->addSuccessors(targets[0]);
                        targets[0]->addPredecessors(source);
                    }
                    this->removeTensor(op1->getOutput(0));
                    del_ops.insert(op);
                }
            }
        };

        do
        {
            size = this->ops.size();
            for (auto &op : ops)
            {
                optimize_transpose_matmul(op);
            }
            for (auto &op : del_ops)
            {
                this->removeOperator(op);
            }
            size_new = this->ops.size();
        } while (size != size_new);
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================

        vector<size_t> offsets(tensors.size());
        for (size_t i = 0; i < tensors.size(); i++)
        {
            offsets[i] = allocator.alloc(tensors[i]->getBytes());
        }
        u_char *ptr = static_cast<u_char *>(allocator.getPtr());
        for (size_t i = 0; i < tensors.size(); i++)
        {
            tensors[i]->setDataBlob(make_ref<BlobObj>(runtime, ptr + offsets[i]));
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini
