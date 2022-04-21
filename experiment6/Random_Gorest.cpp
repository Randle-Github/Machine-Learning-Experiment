#include <bits/stdc++.h>
using namespace std;
#define tree_num 10 // 随机树数量
#define maxn 6000
#define Size 784          // 原始图片大小
#define size 256          // 模糊处理后图片大小
#define max_depth 12      // 规定决策树最大深度
#define learning_num 3000 // 学习样本数
#define test_num 1000     // 测试样本数
inline void read(int &a)  // 快速读入
{
    a = 0;
    char c = getchar();
    while (c < '0' || c > '9')
        c = getchar();
    while (c >= '0' && c <= '9')
    {
        a = (a << 1) + (a << 3) + c - '0';
        c = getchar();
    }
    return;
}

inline double log_2(double x) // 求解以2位底的对数
{
    if (x == 0)
        return -100000.0;
    return log(x) / log(2);
}

struct Fig // 图片
{
    int label;          // 标签
    double vec[size];   // 向量
    double fig[28][28]; // 存储原始图片
    void read_fig()     // 读入一个图片
    {
        read(label);
        for (int i = 0; i < 28; i++) // 读入原始图片
        {
            for (int j = 0; j < 28; j++)
            {
                int x;
                read(x);
                fig[i][j] = double(x);
            }
        }
        int t = sqrt(Size);
        while (t > sqrt(size))
        {
            for (int i = 0; i < t; i++) // 将图像进行模糊化，周围几个灰度值共享一个权值
                for (int j = 0; j < t; j++)
                    fig[i][j] = (fig[i][j] + fig[i][j + 1] + fig[i + 1][j] + fig[i + 1][j + 1]) / 4.0; // 模糊化
            --t;
        }
        int tot = -1;
        for (int i = 0; i < int(sqrt(size)); i++)
            for (int j = 0; j < int(sqrt(size)); j++)
                vec[++tot] = double(int(fig[i][j]) - int(fig[i][j]) % 10); // 最终将Size个特征模糊化为size个特征
    }
} train_fig[60000], test_fig[10000]; // 样本集 测试集

void Read_data() // 读入所有所需图片
{
    freopen("fig_data1.txt", "r", stdin);
    for (int i = 0; i < learning_num; i++)
        train_fig[i].read_fig(); // 读入训练数据
    freopen("CON", "r", stdin);
    freopen("fig_data2.txt", "r", stdin);
    for (int i = 0; i < test_num; i++)
        test_fig[i].read_fig(); // 读入测试数据
    freopen("CON", "r", stdin);
}

struct Hash // 这里设置的Hash结构是为了决策树节点对每一种特征确定划分界限使用的
{
    int node_num; // 标签 节点编号
    double num;   // 对于这个特征的大小
};
bool cmp(Hash a, Hash b) // 自定义比较函数
{
    if (a.num < b.num)
        return true;
    else
        return false;
}

#define root 1                 // 根节点
#define lson node_num << 1     // 左子树
#define rson node_num << 1 | 1 // 右子树

struct Node
{
    int belong;                // 如果是叶子节点，则belong指向所属的类别，否则置-1
    int class_label, node_num; // 这个节点划分特征 节点编号
    double bondary;            // 划分界限，若小于则lson，大于等于则rson
    double information_entropy(vector<int> a, int c, int f, double bond)
    { // 求解对于指定样本、指定划分的特殊标签、指定特征以及指定边界的信息熵
        // information_entropy=sum -pi*log_2(pi)
        double p1 = 0, p2 = 0, p3 = 0, tot = 0;     // 四种情况计算混乱程度
        for (auto i = a.begin(); i != a.end(); i++) // 遍历所有样本计算熵值
        {
            ++tot;
            if (train_fig[*i].vec[f] <= bond)
                p1 += 1.0;
            else
                p2 + 1.0;
        }
        p1 /= tot;
        p2 /= tot;
        return -p1 * log_2(p1) - p2 * log_2(p2);
    }
    void create_node(int p, vector<int> a, Node n[]) //创建决策树节点，输入编号以及划分过来的样本编号
    {
        belong = -1;
        node_num = p;
        if (p > (1 << (max_depth - 1))) // 这个节点已经达到了最大深度，自动设置为叶子节点，投票投出标签
        {
            belong = 0;
            int k[10];
            memset(k, 0, sizeof(k));
            for (auto i = a.begin(); i != a.end(); ++i)
            {
                k[train_fig[*i].label]++; // 标记增加
                if (k[train_fig[*i].label] > k[belong])
                    belong = train_fig[*i].label; // 求得叶子节点出现次数最多的标签
            }
            return;
        }
        belong = train_fig[*a.begin()].label; // 判断所有样本是否属于同一标签
        for (auto i = a.begin(); i != a.end(); i++)
        {
            if (train_fig[*i].label != train_fig[*a.begin()].label) // 节点不纯，不能设置为叶子
            {
                belong = -1;
                break;
            }
        }
        if (belong >= 0) // 已经是叶子节点，belong已经设置成功
            return;
        double max_entropy = -10000000; // 不是叶子节点，需确定最大信息熵的划分形式
        bool choose[size];              // 随机挑选size/10的特征进行划分
        memset(choose, false, sizeof(choose));
        for (int i = 1; i < size / 10; i++)
            choose[rand() % size] = true; // 大致挑选十分之一的特征
        for (int f = 0; f < size; f++)    // 遍历每一个特征，进行划分
        {
            if (!choose[f])
                continue; // 如果这个特征没有被选中则跳过
            Hash hash[maxn];
            int tot = -1;
            for (auto i = a.begin(); i != a.end(); i++)
            {
                hash[++tot].node_num = *i;            // 确定实际对应的训练集编号
                hash[tot].num = train_fig[*i].vec[f]; // 对应的f特征的值
            }
            sort(hash, hash + tot + 1, cmp);  // 内置自定义比较函数，进行一个顺序排序
            for (int i = 0; i < tot + 1; i++) // 此处准备遍历边界，每次使用相邻两个值的中间值作为边界
            {
                if (train_fig[hash[i].node_num].label == train_fig[hash[i + 1].node_num].label)
                    continue; // 如果前后两个属于同一标签，在他们中间划分没有意义
                double entr = information_entropy(a, train_fig[hash[i].node_num].label, f, (hash[i].num + hash[i + 1].num) / 2);
                // 计算a中所有样本，仅仅二分hash[i].node_num所属的那一种标签和其余所有标签，关于f特征，
                // 以其这个特征值与后一个的中间值为划分界限，得到的信息熵。
                if (entr > max_entropy) // 如果以上述形式得到最大的信息熵，则以其为分类标准
                {
                    max_entropy = entr;
                    class_label = f;
                    bondary = (hash[i].num + hash[i + 1].num) / 2;
                }
            }
        }
        vector<int> rson_samples, lson_samples; // 接收右子树的样本
        for (auto i = a.begin(); i != a.end(); i++)
        {
            if (train_fig[*i].vec[class_label] > bondary) // 这些需要进入右子树
                rson_samples.push_back(*i);
            else
                lson_samples.push_back(*i);
        }
        vector<int> temp;
        a.swap(temp);        // 释放内存
        if (bondary < 0.001) // 所有特征相同无法进行分类
        {
            belong = 0;
            int k[10];
            memset(k, 0, sizeof(k));
            for (auto i = a.begin(); i != a.end(); ++i)
            {
                k[train_fig[*i].label]++; // 标记增加
                if (k[train_fig[*i].label] > k[belong])
                    belong = train_fig[*i].label; // 求得叶子节点出现次数最多的标签
            }
            return;
        }
        n[lson].create_node(lson, lson_samples, n); // 创建左子树
        n[rson].create_node(rson, rson_samples, n); // 创建右子树
        return;
    }
    int classification(Fig *p, Node n[]) // 在t节点，将p样本进行分类
    {
        if (belong >= 0) // 已经是叶子节点，直接返回标签
            return belong;
        if (p->vec[class_label] <= bondary) // 进入左子树
            return n[lson].classification(p, n);
        else // 进入右子树
            return n[rson].classification(p, n);
    }
};

struct Tree
{
    Node n[(1 << max_depth) + 1]; // 建立一棵树，并且在此处设计最大深度
    void init()
    {
        vector<int> a; // 根节点需要存储所有样本的序号
        for (int i = 0; i < learning_num; i++)
            a.push_back(i);
        n[root].create_node(root, a, n); // 从根节点开始创建决策树
    }
} tree[tree_num]; // 建立多棵决策树形成随机森林

int vote(Fig *fig) // 进行每棵决策树的投票对fig进行分类
{
    int vote_num[10]; // 0到9的投票结果
    double acc = 0;
    memset(vote_num, 0, sizeof(vote_num)); // 清空
    for (int i = 0; i < tree_num; i++)
        vote_num[tree[i].n[root].classification(fig, tree[i].n)]++; // 从每一棵树的根节点进行分类
    int ans = 0;
    for (int i = 1; i <= 9; i++)
        if (vote_num[i] > vote_num[ans])
            ans = i;
    return ans;
}

int main()
{
    srand(time(NULL));
    Read_data();                       // 读入所有数据
    for (int i = 0; i < tree_num; i++) // 建立tree_num棵决策树
    {
        tree[i].init();
    }
    double acc = 0;
    for (int i = 0; i < test_num; i++) // 测试集分类
    {
        if (vote(&test_fig[i]) == test_fig[i].label) // 进行投票分类
            acc += 1.0;                              // 分类成功
    }
    printf("%lf", acc / test_num * 100);
    cout << '%';
    return 0;
}
