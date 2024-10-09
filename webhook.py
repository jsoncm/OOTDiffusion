import os
import json
import subprocess
from flask import Flask, request

app = Flask(__name__)

@app.route('/ootdhook/', methods=['POST'])
def webhook():
    try:
        # 检查 Content-Type 是否为 application/json
        if request.headers.get('Content-Type') == 'application/json':
            # 获取 GitHub 事件类型
            event = request.headers.get('X-GitHub-Event')
            print(f"Received event: {event}")

            # 解析 JSON 数据
            data = request.json
            
            # 打印收到的完整 JSON 数据
            print("Received payload:")
            # print(json.dumps(data, indent=4))  # 打印 JSON payload，便于调试

            # 仅处理 push 事件
            if event == 'push':
                print("处理push事件...")
                # 确认 'ref' 字段是否存在
                if 'ref' in data:
                    # 检查是否是 master 分支的推送
                    if data['ref'] == 'refs/heads/main':
                        print("Push to main branch detected.")

                        # 进入项目目录
                        os.chdir('/root/project/OOTDiffusion')

                        # 拉取最新代码
                        pull_result = subprocess.run(['git', 'pull'], capture_output=True, text=True)
                        print(pull_result.stdout)  # 输出 git pull 的结果

                        # 重启 streamlit 服务
                        restart_result = subprocess.run(['supervisorctl', 'restart', 'gradio_ootd'], capture_output=True, text=True)
                        print(restart_result.stdout)  # 输出重启的结果 

                        print("Deployment completed successfully.")
                        return 'Deployment completed successfully', 200
                    else:
                        print(f"Push to a different branch: {data['ref']}")
                        return 'Push to a different branch, not master', 200
                else:
                    print("No 'ref' found in the payload")
                    return "Invalid payload: missing 'ref'", 400
            else:
                print(f"Unhandled event: {event}")
                return f"Unhandled event: {event}", 200
        else:
            return 'Unsupported Media Type', 415  # 当 Content-Type 不是 application/json 时返回 415
    except Exception as e:
        print(f"Error: {e}")
        return 'Internal Server Error', 500

#if __name__ == '__main__':
#    app.run(host='0.0.0.0', port=5001, debug=True)
