---@alias HttpMethod '"GET"'|'"POST"'|'"PUT"'|'"DELETE"'|'"PATCH"'|'"HEAD"'|'"OPTIONS"'

---@class FetchOptions
---@field method? HttpMethod HTTP请求方法，默认为"GET"
---@field headers? table<string, string> 请求头，键值对形式
---@field source? string 请求体内容
---@field timeout? number 超时时间（秒）
---@field [string] any 其他可能的选项（虽然curl不一定支持）

local function shell_quote(value)
    return "'" .. tostring(value):gsub("'", "'\"'\"'") .. "'"
end

---使用curl发送HTTP请求并获取响应
---@param url string 请求的URL
---@param op FetchOptions 请求选项
---@return integer? status_code HTTP状态码，失败时为nil
---@return string|nil response_body 响应体字符串，失败时为错误信息
local function fetch_text(url, op)
    -- 超时是必须的，不是可选项：
    -- 输入法是 TSF 服务，会被加载进每一个应用程序的进程里，下面的 io.popen +
    -- read("*a") 是同步阻塞调用，直接卡住当前应用的 UI 线程。后端一慢（比如模型
    -- 正在加载）而 curl 又不设超时，表现就是整台机器像死机一样敲不进字。
    -- 原来的代码声明了 op.timeout 却从没用过，实测导致过必须重启。
    local timeout = op.timeout or 2
    local command = string.format(
        "curl -s --connect-timeout 1 --max-time %s -w '\n%%{http_code}' --request %s --url %s",
        shell_quote(timeout), shell_quote(op.method or "GET"), shell_quote(url))

    -- 添加 headers
    if op.headers then
        for k, v in pairs(op.headers) do
            command = command .. string.format(" --header %s", shell_quote(k .. ": " .. v))
        end
    end

    -- 添加请求体
    if op.source then
        command = command .. string.format(" --data %s", shell_quote(op.source))
    end

    -- 执行 curl 命令
    local handle = io.popen(command)
    if not handle then
        return nil, "无法执行 curl 命令"
    end

    local output = handle:read("*a")
    handle:close()

    -- 解析输出，分离响应体和状态码
    local last_newline = output:find("\n[0-9][0-9][0-9]$")
    if last_newline then
        local response_body = output:sub(1, last_newline - 1)
        local status_code = tonumber(output:sub(last_newline + 1))
        return status_code, response_body
    else
        return nil, "无效的响应格式"
    end
end

return fetch_text
