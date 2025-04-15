/******************************************************************************

Welcome to GDB Online.
GDB online is an online compiler and debugger tool for C, C++, Python, Java, PHP, Ruby, Perl,
C#, OCaml, VB, Swift, Pascal, Fortran, Haskell, Objective-C, Assembly, HTML, CSS, JS, SQLite, Prolog.
Code, Compile, Run and Debug online from anywhere in world.

*******************************************************************************/


#include <iostream>
#include <vector>
#include <stack>
#include <unordered_map>

using namespace std;


struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode* parent;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr), parent(nullptr) {}
};

// 非遞歸查找從根到目標節點的路徑方向
bool findDirectionPathNonRecursive(TreeNode* root, int target, vector<int>& path) {
    if (!root) return false;

    stack<pair<TreeNode*, vector<int>>> s;
    s.push({root, {}});

    while (!s.empty()) {
        auto [node, currentPath] = s.top();
        s.pop();

        if (node->val == target) {
            path = currentPath;
            return true;
        }

        if (node->right) {
            vector<int> newPath = currentPath;
            newPath.push_back(2); // 向右
            s.push({node->right, newPath});
        }

        if (node->left) {
            vector<int> newPath = currentPath;
            newPath.push_back(1); // 向左
            s.push({node->left, newPath});
        }
    }

    return false;
}

// 查找兩個節點之間的路徑（帶詳細錯誤信息）
pair<vector<int>, string> findPathBetweenNodesWithDirections(TreeNode* root, int node1, int node2) {
    vector<int> path1, path2;
    string errorMsg;

    // 檢查空樹
    if (!root) {
        return {{}, "empty no node in tree"};
    }

    // 查找路徑
    bool found1 = findDirectionPathNonRecursive(root, node1, path1);
    bool found2 = findDirectionPathNonRecursive(root, node2, path2);

    // 處理節點不存在的情況
    if (!found1 && !found2) {
        errorMsg = "node " + to_string(node1) + " and " + to_string(node2) + " miss";
        return {{}, errorMsg};
    } else if (!found1) {
        errorMsg = "node " + to_string(node1) + " miss";
        return {{}, errorMsg};
    } else if (!found2) {
        errorMsg = "node " + to_string(node2) + " miss";
        return {{}, errorMsg};
    }

    // 尋找共同前綴長度
    int commonLength = 0;
    while (commonLength < path1.size() && 
           commonLength < path2.size() && 
           path1[commonLength] == path2[commonLength]) {
        commonLength++;
    }

    // 構建最終路徑
    vector<int> result;

    // 從node1到LCA的路徑（需要往父節點走，用0表示）
    for (int i = path1.size() - 1; i >= commonLength; --i) {
        result.push_back(0); // 往父節點
    }

    // 從LCA到node2的路徑（使用原來的方向）
    for (int i = commonLength; i < path2.size(); ++i) {
        result.push_back(path2[i]);
    }

    return {result, ""};
}

// 創建樹並設置父節點指針
TreeNode* buildTreeWithParent(const vector<int>& values, int start, int end, TreeNode* parent = nullptr) {
    if (start > end) return nullptr;

    int mid = start + (end - start) / 2;
    TreeNode* node = new TreeNode(values[mid]);
    node->parent = parent;

    node->left = buildTreeWithParent(values, start, mid - 1, node);
    node->right = buildTreeWithParent(values, mid + 1, end, node);

    return node;
}

// 打印路徑結果
void printPathResult(const pair<vector<int>, string>& result, int node1, int node2) {
    if (!result.second.empty()) {
        cout << "mistake: " << result.second << endl;
        return;
    }

    cout << "from " << node1 << " to " << node2 << " path: ";
    for (int dir : result.first) {
        switch(dir) {
            case 0: cout << "↑ "; break;
            case 1: cout << "← "; break;
            case 2: cout << "→ "; break;
        }
    }
   for (int dir : result.first) {
         cout <<"."<<dir;
    }
    cout << endl;
}

// 測試案例
void runTestCases() {
    // 1. 完美平衡樹
//        4
//       / \
//      2   6
//     / \ / \
//    1  3 5 7
    cout << "=== 完美平衡樹測試 ===" << endl;
    vector<int> balancedValues = {1, 2, 3, 4, 5, 6, 7};
    TreeNode* balancedTree = buildTreeWithParent(balancedValues, 0, balancedValues.size() - 1);
    
    auto test1 = findPathBetweenNodesWithDirections(balancedTree, 1, 7);
    printPathResult(test1, 1, 7);  // 預期: ↑ ↑ → →
    
    auto test2 = findPathBetweenNodesWithDirections(balancedTree, 3, 5);
    printPathResult(test2, 3, 5);  // 預期: ↑ ↑ → ←
    
    auto test3 = findPathBetweenNodesWithDirections(balancedTree, 4, 4);
    printPathResult(test3, 4, 4);  // 預期: (空路徑)
    
    auto test4 = findPathBetweenNodesWithDirections(balancedTree, 1, 9);
    printPathResult(test4, 1, 9);  // 預期: 錯誤
    
    cout << endl;

    // 2. 單節點樹
    cout << "=== 單節點樹測試 ===" << endl;
    TreeNode* singleNodeTree = new TreeNode(1);
    
    auto test5 = findPathBetweenNodesWithDirections(singleNodeTree, 1, 1);
    printPathResult(test5, 1, 1);  // 預期: (空路徑)
    
    auto test6 = findPathBetweenNodesWithDirections(singleNodeTree, 1, 2);
    printPathResult(test6, 1, 2);  // 預期: 錯誤
    
    cout << endl;

    // 3. 空樹
    cout << "=== 空樹測試 ===" << endl;
    TreeNode* emptyTree = nullptr;
    
    auto test7 = findPathBetweenNodesWithDirections(emptyTree, 1, 2);
    printPathResult(test7, 1, 2);  // 預期: 錯誤
    
    cout << endl;

    // 4. 複雜案例
    cout << "=== 複雜結構測試 ===" << endl;
    //    1
    //   / \
    //  2   3
    //   \
    //    4
    TreeNode* complexTree = new TreeNode(1);
    complexTree->left = new TreeNode(2);
    complexTree->right = new TreeNode(3);
    complexTree->left->parent = complexTree;
    complexTree->right->parent = complexTree;
    complexTree->left->right = new TreeNode(4);
    complexTree->left->right->parent = complexTree->left;
    
    auto test8 = findPathBetweenNodesWithDirections(complexTree, 4, 3);
    printPathResult(test8, 4, 3);  // 預期: ↑ ↑ →
    
    auto test9 = findPathBetweenNodesWithDirections(complexTree, 2, 4);
    printPathResult(test9, 2, 4);  // 預期: →
}

int main() {
    runTestCases();
    return 0;
}
