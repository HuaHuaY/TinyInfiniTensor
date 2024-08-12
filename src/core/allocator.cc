#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================

        used += size;
        size_t res = SIZE_MAX;
        FreeBlock *p = &head;
        while (p->next)
        {
            if (p->next->size > size)
            {
                res = p->next->addr;
                p->next->addr += size;
                break;
            }
            else if (p->next->size == size)
            {
                FreeBlock *tmp = p->next;
                res = tmp->addr;
                p->next = tmp->next;
                tmp->next = nullptr;
                delete tmp;
                break;
            }
            else if (p->next->addr + p->next->size == peak)
            {
                peak -= p->next->size;
                delete p->next;
                p->next = nullptr;
                break;
            }
            p = p->next;
        }
        if (res == SIZE_MAX)
        {
            res = peak;
            peak += size;
        }

        return res;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================

        used -= size;
        FreeBlock *p = &head;
        while (p)
        {
            if (!p->next)
            {
                if (p != &head && p->addr + p->size == addr)
                {
                    p->size += size;
                    return;
                }
                p->next = new FreeBlock(addr, size);
                return;
            }
            if (addr + size > p->next->addr)
            {
                p = p->next;
                continue;
            }

            if (addr + size == p->next->addr)
            {
                p->next->addr = addr;
                p->next->size += size;
            }
            else
            {
                FreeBlock *tmp = p->next;
                p->next = new FreeBlock(addr, size);
                p->next->next = tmp;
            }
            if (p != &head && p->addr + p->size == p->next->addr)
            {
                FreeBlock *tmp = p->next;
                p->size += tmp->size;
                p->next = tmp->next;
                tmp->next = nullptr;
                delete tmp;
            }
            return;
        }
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
