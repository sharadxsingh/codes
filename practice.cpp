2. Quick Sort
cpp
Copy
Edit
#include <iostream>
using namespace std;

int partition(int arr[], int low, int high) {
    int pivot = arr[high], i = low - 1;
    for (int j = low; j < high; j++)
        if (arr[j] < pivot)
            swap(arr[++i], arr[j]);
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int p = partition(arr, low, high);
        quickSort(arr, low, p - 1);
        quickSort(arr, p + 1, high);
    }
}

int main() {
    int arr[] = {10, 7, 8, 9, 1, 5}, n = 6;
    quickSort(arr, 0, n - 1);
    cout << "Sorted array: ";
    for (int i = 0; i < n; i++) cout << arr[i] << " ";
    return 0;
}
3. Merge Sort
cpp
Copy
Edit
#include <iostream>
using namespace std;

void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1, n2 = r - m;
    int L[n1], R[n2];
    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

void mergeSort(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

int main() {
    int arr[] = {12, 11, 13, 5, 6, 7}, n = 6;
    mergeSort(arr, 0, n - 1);
    cout << "Sorted array: ";
    for (int i = 0; i < n; i++) cout << arr[i] << " ";
    return 0;
}
4. Finding Max & Min (Divide and Conquer)
cpp
Copy
Edit
#include <iostream>
using namespace std;

void findMaxMin(int arr[], int low, int high, int &min, int &max) {
    if (low == high) {
        max = min = arr[low];
    } else if (high == low + 1) {
        if (arr[low] < arr[high]) {
            min = arr[low]; max = arr[high];
        } else {
            min = arr[high]; max = arr[low];
        }
    } else {
        int mid = (low + high) / 2;
        int min1, max1, min2, max2;
        findMaxMin(arr, low, mid, min1, max1);
        findMaxMin(arr, mid + 1, high, min2, max2);
        min = (min1 < min2) ? min1 : min2;
        max = (max1 > max2) ? max1 : max2;
    }
}

int main() {
    int arr[] = {100, 4, 200, 1, 55}, min, max;
    findMaxMin(arr, 0, 4, min, max);
    cout << "Min: " << min << ", Max: " << max << endl;
    return 0;
}
(continued in next message due to length...)







You said:
Continue 
ChatGPT said:
Continuing with the rest of the experiments:

5. BFS and DFS in Graph
cpp
Copy
Edit
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

vector<vector<int>> adj;
vector<bool> visited;

void BFS(int start) {
    queue<int> q;
    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int node = q.front(); q.pop();
        cout << node << " ";
        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}

void DFS(int node) {
    visited[node] = true;
    cout << node << " ";
    for (int neighbor : adj[node])
        if (!visited[neighbor]) DFS(neighbor);
}

int main() {
    int nodes = 5;
    adj.assign(nodes, {});
    visited.assign(nodes, false);

    adj[0] = {1, 2};
    adj[1] = {0, 3};
    adj[2] = {0, 4};
    adj[3] = {1};
    adj[4] = {2};

    cout << "BFS: ";
    BFS(0);
    fill(visited.begin(), visited.end(), false);
    cout << "\nDFS: ";
    DFS(0);
    return 0;
}
6. Prim’s and Kruskal’s Algorithm
Kruskal’s Algorithm

cpp
Copy
Edit
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Edge {
    int u, v, weight;
    bool operator<(Edge const &other) { return weight < other.weight; }
};

int find(int v, vector<int> &parent) {
    if (parent[v] == v) return v;
    return parent[v] = find(parent[v], parent);
}

void unionSet(int a, int b, vector<int> &parent) {
    a = find(a, parent);
    b = find(b, parent);
    if (a != b) parent[b] = a;
}

int main() {
    int V = 4;
    vector<Edge> edges = {{0, 1, 10}, {0, 2, 6}, {0, 3, 5}, {1, 3, 15}, {2, 3, 4}};
    sort(edges.begin(), edges.end());

    vector<int> parent(V);
    for (int i = 0; i < V; i++) parent[i] = i;

    cout << "Edges in MST:\n";
    for (Edge e : edges) {
        if (find(e.u, parent) != find(e.v, parent)) {
            cout << e.u << " - " << e.v << " : " << e.weight << endl;
            unionSet(e.u, e.v, parent);
        }
    }
    return 0;
}
7. Knapsack using Greedy (Fractional Knapsack)
cpp
Copy
Edit
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Item {
    int value, weight;
    bool operator<(Item const &a) const {
        return (double)value / weight > (double)a.value / a.weight;
    }
};

double fractionalKnapsack(int W, vector<Item> &items) {
    sort(items.begin(), items.end());
    double totalValue = 0.0;
    for (auto &item : items) {
        if (W >= item.weight) {
            W -= item.weight;
            totalValue += item.value;
        } else {
            totalValue += item.value * ((double)W / item.weight);
            break;
        }
    }
    return totalValue;
}

int main() {
    vector<Item> items = {{60, 10}, {100, 20}, {120, 30}};
    int W = 50;
    cout << "Max value in Knapsack = " << fractionalKnapsack(W, items) << endl;
    return 0;
}
8. Job Sequencing Problem + 0/1 Knapsack (DP)
cpp
Copy
Edit
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Job {
    char id;
    int deadline, profit;
};

bool compare(Job a, Job b) {
    return a.profit > b.profit;
}

void jobScheduling(vector<Job> jobs) {
    sort(jobs.begin(), jobs.end(), compare);
    int n = jobs.size();
    vector<bool> slot(n, false);
    vector<char> result(n);

    for (auto job : jobs) {
        for (int j = min(n, job.deadline) - 1; j >= 0; j--) {
            if (!slot[j]) {
                slot[j] = true;
                result[j] = job.id;
                break;
            }
        }
    }

    cout << "Scheduled Jobs: ";
    for (int i = 0; i < n; i++)
        if (slot[i]) cout << result[i] << " ";
    cout << endl;
}

int main() {
    vector<Job> jobs = {{'a', 2, 100}, {'b', 1, 19}, {'c', 2, 27}, {'d', 1, 25}, {'e', 3, 15}};
    jobScheduling(jobs);
    return 0;
}
9. Matrix Chain Multiplication (DP)
cpp
Copy
Edit
#include <iostream>
#include <climits>
using namespace std;

int matrixChainOrder(int p[], int n) {
    int dp[n][n];
    for (int i = 1; i < n; i++) dp[i][i] = 0;

    for (int len = 2; len < n; len++) {
        for (int i = 1; i < n - len + 1; i++) {
            int j = i + len - 1;
            dp[i][j] = INT_MAX;
            for (int k = i; k < j; k++) {
                dp[i][j] = min(dp[i][j],
                    dp[i][k] + dp[k + 1][j] + p[i - 1] * p[k] * p[j]);
            }
        }
    }
    return dp[1][n - 1];
}

int main() {
    int arr[] = {1, 2, 3, 4};
    int n = sizeof(arr) / sizeof(arr[0]);
    cout << "Minimum multiplications: " << matrixChainOrder(arr, n) << endl;
    return 0;
}
Next Message: Experiments 10 to 12 (Graph Coloring, LCS, 0/1 Knapsack Branch & Bound)
You said:
Continue 
ChatGPT said:
Continuing with the remaining DAA experiments:

10. Graph Coloring (Backtracking)
cpp
Copy
Edit
#include <iostream>
using namespace std;

#define V 4

bool isSafe(int v, bool graph[V][V], int color[], int c) {
    for (int i = 0; i < V; i++)
        if (graph[v][i] && color[i] == c)
            return false;
    return true;
}

bool graphColoringUtil(bool graph[V][V], int m, int color[], int v) {
    if (v == V) return true;

    for (int c = 1; c <= m; c++) {
        if (isSafe(v, graph, color, c)) {
            color[v] = c;
            if (graphColoringUtil(graph, m, color, v + 1)) return true;
            color[v] = 0;
        }
    }

    return false;
}

bool graphColoring(bool graph[V][V], int m) {
    int color[V] = {0};
    return graphColoringUtil(graph, m, color, 0);
}

int main() {
    bool graph[V][V] = {
        {0, 1, 1, 1},
        {1, 0, 1, 0},
        {1, 1, 0, 1},
        {1, 0, 1, 0}
    };
    int m = 3;
    if (graphColoring(graph, m))
        cout << "Solution exists with " << m << " colors\n";
    else
        cout << "No solution\n";
    return 0;
}
11. Longest Common Subsequence (LCS)
cpp
Copy
Edit
#include <iostream>
#include <string>
using namespace std;

int lcs(string X, string Y) {
    int m = X.length(), n = Y.length();
    int dp[m + 1][n + 1];

    for (int i = 0; i <= m; i++)
        for (int j = 0; j <= n; j++) {
            if (i == 0 || j == 0)
                dp[i][j] = 0;
            else if (X[i - 1] == Y[j - 1])
                dp[i][j] = 1 + dp[i - 1][j - 1];
            else
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
        }

    return dp[m][n];
}

int main() {
    string X = "AGGTAB", Y = "GXTXAYB";
    cout << "Length of LCS: " << lcs(X, Y) << endl;
    return 0;
}
12. 0/1 Knapsack using Branch and Bound
cpp
Copy
Edit
#include <iostream>
#include <queue>
using namespace std;

struct Item {
    int weight, value;
};

struct Node {
    int level, profit, bound, weight;
};

bool cmp(Item a, Item b) {
    return (double)a.value / a.weight > (double)b.value / b.weight;
}

int bound(Node u, int n, int W, Item arr[]) {
    if (u.weight >= W) return 0;
    int profit_bound = u.profit;
    int j = u.level + 1, totweight = u.weight;

    while (j < n && totweight + arr[j].weight <= W) {
        totweight += arr[j].weight;
        profit_bound += arr[j].value;
        j++;
    }

    if (j < n)
        profit_bound += (W - totweight) * arr[j].value / arr[j].weight;

    return profit_bound;
}

int knapsack(int W, Item arr[], int n) {
    sort(arr, arr + n, cmp);
    queue<Node> Q;
    Node u, v;
    u.level = -1, u.profit = u.weight = 0;
    Q.push(u);

    int maxProfit = 0;
    while (!Q.empty()) {
        u = Q.front(); Q.pop();
        if (u.level == n - 1) continue;

        v.level = u.level + 1;
        v.weight = u.weight + arr[v.level].weight;
        v.profit = u.profit + arr[v.level].value;

        if (v.weight <= W && v.profit > maxProfit)
            maxProfit = v.profit;

        v.bound = bound(v, n, W, arr);
        if (v.bound > maxProfit)
            Q.push(v);

        v.weight = u.weight;
        v.profit = u.profit;
        v.bound = bound(v, n, W, arr);
        if (v.bound > maxProfit)
            Q.push(v);
    }
    return maxProfit;
}

int main() {
    Item arr[] = {{10, 60}, {20, 100}, {30, 120}};
    int W = 50, n = sizeof(arr) / sizeof(arr[0]);
    cout << "Maximum profit: " << knapsack(W, arr, n) << endl;
    return 0;
}

✅ Prim’s Algorithm (Minimum Spanning Tree)
cpp
Copy
Edit
#include <iostream>
#include <climits>
using namespace std;

#define V 5

int minKey(int key[], bool mstSet[]) {
    int min = INT_MAX, min_index;

    for (int v = 0; v < V; v++)
        if (!mstSet[v] && key[v] < min)
            min = key[v], min_index = v;

    return min_index;
}

void printMST(int parent[], int graph[V][V]) {
    cout << "Edge \tWeight\n";
    for (int i = 1; i < V; i++)
        cout << parent[i] << " - " << i << " \t" << graph[i][parent[i]] << endl;
}

void primMST(int graph[V][V]) {
    int parent[V]; // Store MST
    int key[V]; // Weights
    bool mstSet[V]; // Included in MST

    for (int i = 0; i < V; i++)
        key[i] = INT_MAX, mstSet[i] = false;

    key[0] = 0;     
    parent[0] = -1; 

    for (int count = 0; count < V - 1; count++) {
        int u = minKey(key, mstSet);

        mstSet[u] = true;

        for (int v = 0; v < V; v++)
            if (graph[u][v] && !mstSet[v] && graph[u][v] < key[v])
                parent[v] = u, key[v] = graph[u][v];
    }

    printMST(parent, graph);
}

int main() {
    int graph[V][V] = {
        {0, 2, 0, 6, 0},
        {2, 0, 3, 8, 5},
        {0, 3, 0, 0, 7},
        {6, 8, 0, 0, 9},
        {0, 5, 7, 9, 0}
    };

    primMST(graph);
    return 0;
}

✅ 0/1 Knapsack using Dynamic Programming
cpp
Copy
Edit
#include <iostream>
using namespace std;

int knapsack(int weights[], int values[], int n, int W) {
    int dp[n + 1][W + 1];

    // Build table dp[][] in bottom up manner
    for (int i = 0; i <= n; i++) {
        for (int w = 0; w <= W; w++) {
            // Base case: 0 items or 0 capacity
            if (i == 0 || w == 0)
                dp[i][w] = 0;

            // If weight of current item <= capacity
            else if (weights[i - 1] <= w)
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]],
                               dp[i - 1][w]);

            // If weight exceeds capacity, exclude item
            else
                dp[i][w] = dp[i - 1][w];
        }
    }

    return dp[n][W];
}

int main() {
    int values[] = {60, 100, 120};
    int weights[] = {10, 20, 30};
    int W = 50;
    int n = sizeof(values) / sizeof(values[0]);

    cout << "Maximum value in 0/1 Knapsack = " << knapsack(weights, values, n, W) << endl;

    return 0;
}

✅ Binary Search (Already Given, but here's again for clarity)
cpp
Copy
Edit
#include <iostream>
using namespace std;

int binarySearch(int arr[], int n, int key) {
    int low = 0, high = n - 1;

    while (low <= high) {
        int mid = (low + high) / 2;

        if (arr[mid] == key)
            return mid;
        else if (arr[mid] < key)
            low = mid + 1;
        else
            high = mid - 1;
    }

    return -1;
}

int main() {
    int arr[] = {2, 4, 6, 8, 10};
    int n = 5;
    int key = 6;
    int result = binarySearch(arr, n, key);

    if (result != -1)
        cout << "Element found at index " << result << endl;
    else
        cout << "Element not found" << endl;

    return 0;
}
