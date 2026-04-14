import requests

class InterfaceClient:
    def __init__(self, ip="127.0.0.1", port="8000"):
        self.robot_ip = ip
        self.robot_port = port
        self.head_list = ["config", "state", "obs", "action"]
    
    def send_message(self, head:str, content:any) -> dict:
        '''index: "config", "state", "obs", "action"'''
        try:
            if head not in self.head_list:
                raise e
            address = "http://" + self.robot_ip + ":" + self.robot_port + "/message"
            msg ={
                "head":head,
                "content":content,
            }
            resp = requests.post(
                address,
                json=msg,
                timeout=3
            )
            reply = resp.json()
            print(f"[interface client] 已发送给 robot /message: {msg}")
            
            if reply is not None:
                print(f"[interface client] 收到来自 robot 的回复: {reply}")
                return reply
            else:
                return None

        except requests.RequestException as e:
            print(f"[interface client]] 发送失败: {e}")

