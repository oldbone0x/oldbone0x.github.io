"use strict";(self.webpackChunkoldbone_site_docusaurus=self.webpackChunkoldbone_site_docusaurus||[]).push([[4686],{2315:(e,s,n)=>{n.r(s),n.d(s,{assets:()=>o,contentTitle:()=>t,default:()=>u,frontMatter:()=>l,metadata:()=>c,toc:()=>d});var i=n(5893),r=n(1151);const l={},t=void 0,c={id:"offline-install/k3s",title:"k3s",description:"\u672c\u6587\u5185\u5bb9",source:"@site/docs/offline-install/2-k3s.md",sourceDirName:"offline-install",slug:"/offline-install/k3s",permalink:"/docs/offline-install/k3s",draft:!1,unlisted:!1,tags:[],version:"current",sidebarPosition:2,frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Docker",permalink:"/docs/offline-install/docker"},next:{title:"MySQL 8",permalink:"/docs/offline-install/mysql-8"}},o={},d=[{value:"\u672c\u6587\u5185\u5bb9",id:"\u672c\u6587\u5185\u5bb9",level:2},{value:"\u90e8\u7f72\u80cc\u666f",id:"\u90e8\u7f72\u80cc\u666f",level:2},{value:"\u4ec0\u4e48\u662fK3S",id:"\u4ec0\u4e48\u662fk3s",level:2},{value:"\u5b89\u88c5\u5305\u51c6\u5907",id:"\u5b89\u88c5\u5305\u51c6\u5907",level:2},{value:"K3S\u5b89\u88c5",id:"k3s\u5b89\u88c5",level:2},{value:"\u79c1\u6709\u955c\u50cf\u4ed3\u5e93\u914d\u7f6e",id:"\u79c1\u6709\u955c\u50cf\u4ed3\u5e93\u914d\u7f6e",level:2}];function a(e){const s={a:"a",blockquote:"blockquote",br:"br",code:"code",h2:"h2",li:"li",ol:"ol",p:"p",pre:"pre",ul:"ul",...(0,r.a)(),...e.components};return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(s.h2,{id:"\u672c\u6587\u5185\u5bb9",children:"\u672c\u6587\u5185\u5bb9"}),"\n",(0,i.jsxs)(s.ul,{children:["\n",(0,i.jsx)(s.li,{children:"\u901a\u8fc7binary\u65b9\u5f0f\u90e8\u7f72\u8f7b\u91cf\u7ea7kubernertes(K3S)"}),"\n",(0,i.jsx)(s.li,{children:"\u914d\u7f6eK3S\u4f7f\u7528\u79c1\u6709\u955c\u50cf\u4ed3\u5e93"}),"\n"]}),"\n",(0,i.jsx)(s.h2,{id:"\u90e8\u7f72\u80cc\u666f",children:"\u90e8\u7f72\u80cc\u666f"}),"\n",(0,i.jsxs)(s.ul,{children:["\n",(0,i.jsx)(s.li,{children:"\u6240\u5728\u670d\u52a1\u5668\u8bbf\u95ee\u4e0d\u5230\u5916\u7f51\uff0c\u53ea\u80fd\u79bb\u7ebf\u90e8\u7f72"}),"\n",(0,i.jsx)(s.li,{children:"\u670d\u52a1\u5668OS\u7248\u672c\u4e3aCentOS 7.9"}),"\n",(0,i.jsxs)(s.li,{children:["\u5df2\u7ecf\u642d\u5efa\u597d\u79c1\u6709docker\u955c\u50cf\u4ed3\u5e93\u3002\uff08\u5982\u6ca1\u6709\u53ef\u79fb\u6b65 ",(0,i.jsx)(s.a,{href:"/docs/offline-install/docker",children:"\u79bb\u7ebf\u73af\u5883\u90e8\u7f72docker\u53ca\u79c1\u6709\u955c\u50cf\u4ed3\u5e93"}),"\uff09"]}),"\n"]}),"\n",(0,i.jsx)(s.h2,{id:"\u4ec0\u4e48\u662fk3s",children:"\u4ec0\u4e48\u662fK3S"}),"\n",(0,i.jsxs)(s.blockquote,{children:["\n",(0,i.jsx)(s.p,{children:"K3s \u662f\u8f7b\u91cf\u7ea7\u7684 Kubernetes\u3002K3s \u6613\u4e8e\u5b89\u88c5\uff0c\u800c\u4e14\u4ec5\u9700\u8981 Kubernetes \u5185\u5b58\u7684\u4e00\u534a\uff0c\u6240\u6709\u7ec4\u4ef6\u90fd\u5728\u4e00\u4e2a\u5c0f\u4e8e 100 MB \u7684\u4e8c\u8fdb\u5236\u6587\u4ef6\u4e2d\u3002"}),"\n"]}),"\n",(0,i.jsx)(s.h2,{id:"\u5b89\u88c5\u5305\u51c6\u5907",children:"\u5b89\u88c5\u5305\u51c6\u5907"}),"\n",(0,i.jsxs)(s.ol,{children:["\n",(0,i.jsxs)(s.li,{children:[(0,i.jsx)(s.code,{children:"k3s-airgap-images-amd64.tar.gz"}),". K3S\u955c\u50cf\u5305\uff0c\u4e0b\u8f7d\u5730\u5740 ",(0,i.jsx)(s.a,{href:"https://github.com/k3s-io/k3s/releases",children:"https://github.com/k3s-io/k3s/releases"})," . \u7531\u4e8e\u670d\u52a1\u5668\u7684CPU\u662fX86_64\u67b6\u6784\uff0c\u6240\u4ee5\u9009\u62e9 ",(0,i.jsx)(s.code,{children:"amd64"}),"\u7248\u672c."]}),"\n",(0,i.jsxs)(s.li,{children:[(0,i.jsx)(s.code,{children:"k3s"}),".K3S \u53ef\u6267\u884c\u6587\u4ef6. \u4e0b\u8f7d\u5730\u5740 ",(0,i.jsx)(s.a,{href:"https://github.com/k3s-io/k3s/releases",children:"https://github.com/k3s-io/k3s/releases"})]}),"\n",(0,i.jsxs)(s.li,{children:[(0,i.jsx)(s.code,{children:"install.sh"}),". K3S\u5b89\u88c5\u811a\u672c.\u4e0b\u8f7d\u5730\u5740 ",(0,i.jsx)(s.a,{href:"https://get.k3s.io",children:"https://get.k3s.io"})]}),"\n",(0,i.jsxs)(s.li,{children:[(0,i.jsx)(s.code,{children:"k3s-selinux-1.4-1.el7.noarch.rpm"}),". \u975e\u5fc5\u987b\uff0c\u53ea\u6709\u5f53\u4f60\u7684\u670d\u52a1\u5668\u9700\u8981SELinux\u65f6\u624d\u9700\u8981\u5b89\u88c5\u3002\u4e0b\u8f7d\u5730\u5740",(0,i.jsx)(s.a,{href:"https://github.com/k3s-io/k3s-selinux/releases/tag/v1.4.stable.1",children:"https://github.com/k3s-io/k3s-selinux/releases/tag/v1.4.stable.1"})]}),"\n"]}),"\n",(0,i.jsx)(s.h2,{id:"k3s\u5b89\u88c5",children:"K3S\u5b89\u88c5"}),"\n",(0,i.jsxs)(s.p,{children:["\u5c06\u5b89\u88c5\u5305\u4e0a\u4f20\u5230\u76ee\u6807\u670d\u52a1\u5668,\u5e76\u653e\u5728\u540c\u4e00\u4e2a\u76ee\u5f55",(0,i.jsx)(s.br,{}),"\n","\u5b89\u88c5SELinux\u652f\u6301\uff08\u5982\u4e0d\u9700\u8981\u53ef\u8df3\u8fc7\u6b64\u6b65\u9aa4\uff09",(0,i.jsx)(s.br,{}),"\n","\u5047\u5982\u51fa\u73b0\u7f3a\u5c11container-selinux \u4f9d\u8d56\u7684\u62a5\u9519\uff0c\u4e0b\u8f7d\u5b89\u88c5 ",(0,i.jsx)(s.a,{href:"http://mirror.centos.org/centos/7/extras/x86_64/Packages/container-selinux-2.107-3.el7.noarch.rpm",children:"container-selinux-2.107-3.el7.noarch.rpm"})]}),"\n",(0,i.jsx)(s.pre,{children:(0,i.jsx)(s.code,{children:"$ sudo rpm -ivh container-selinux-2.107-3.el7.noarch.rpm\n$ sudo rpm -ivh k3s-selinux-1.4-1.el7.noarch.rpm \n"})}),"\n",(0,i.jsx)(s.p,{children:"\u5173\u95edfirewalld"}),"\n",(0,i.jsx)(s.pre,{children:(0,i.jsx)(s.code,{children:"$ sudo systemctl disable firewalld --now\n"})}),"\n",(0,i.jsx)(s.p,{children:"\u5982\u679c\u4f60\u5e0c\u671b\u4fdd\u6301firewalld\u7684\u5f00\u542f\u72b6\u6001\uff0c\u90a3\u9700\u8981\u4e3aK3S\u52a0\u51e0\u6761\u89c4\u5219"}),"\n",(0,i.jsx)(s.pre,{children:(0,i.jsx)(s.code,{children:"$ firewall-cmd --permanent --add-port=6443/tcp #apiserver\n$ firewall-cmd --permanent --zone=trusted --add-source=10.42.0.0/16 #pods\n$ firewall-cmd --permanent --zone=trusted --add-source=10.43.0.0/16 #services\n$ firewall-cmd --reload\n"})}),"\n",(0,i.jsx)(s.p,{children:"\u5b89\u88c5K3S"}),"\n",(0,i.jsx)(s.pre,{children:(0,i.jsx)(s.code,{children:"$ sudo mkdir -p /var/lib/rancher/k3s/agent/images/\n$ sudo cp ./k3s-airgap-images-amd64.tar.gz /var/lib/rancher/k3s/agent/images/\n$ sudo cp ./k3s /usr/local/bin/\n$ sudo chmod +x /usr/local/bin/k3s\n$ INSTALL_K3S_SELINUX_WARN=true INSTALL_K3S_SKIP_DOWNLOAD=true ./install.sh --write-kubeconfig-mode=644\n[INFO]  Skipping k3s download and verify\n[INFO]  Skipping installation of SELinux RPM\n[WARN]  Failed to find the k3s-selinux policy, please install:\n    yum install -y container-selinux\n    yum install -y https://rpm.rancher.io/k3s/stable/common/centos/7/noarch/\n\n[INFO]  Creating /usr/local/bin/kubectl symlink to k3s\n[INFO]  Creating /usr/local/bin/crictl symlink to k3s\n[INFO]  Skipping /usr/local/bin/ctr symlink to k3s, command exists in PATH at /bin/ctr\n[INFO]  Creating killall script /usr/local/bin/k3s-killall.sh\n[INFO]  Creating uninstall script /usr/local/bin/k3s-uninstall.sh\n[INFO]  env: Creating environment file /etc/systemd/system/k3s.service.env\n[INFO]  systemd: Creating service file /etc/systemd/system/k3s.service\n[INFO]  systemd: Enabling k3s unit\nCreated symlink from /etc/systemd/system/multi-user.target.wants/k3s.service to /etc/systemd/system/k3s.service.\n[INFO]  systemd: Starting k3s\n"})}),"\n",(0,i.jsx)(s.p,{children:"\u53c2\u6570\u8bf4\u660e:"}),"\n",(0,i.jsxs)(s.blockquote,{children:["\n",(0,i.jsxs)(s.p,{children:["INSTALL_K3S_SELINUX_WARN=true: \u5f53\u7f3a\u5c11",(0,i.jsx)(s.code,{children:"k3s-selinux"}),"\u4f9d\u8d56\u65f6\uff0c\u4e5f\u4f1a\u7ee7\u7eed\u5b89\u88c5",(0,i.jsx)(s.br,{}),"\n","INSTALL_K3S_SKIP_DOWNLOAD=true: \u8df3\u8fc7\u4e0b\u8f7dk3s\u5b89\u88c5\u5305",(0,i.jsx)(s.br,{}),"\n","--write-kubeconfig-mode=644: \u4fee\u6539k3s\u914d\u7f6e\u6587\u4ef6\u7684\u6743\u9650\uff0c\u907f\u514d\u666e\u901a\u7528\u6237\u65e0\u6cd5\u4f7f\u7528kubectl\u7684\u60c5\u51b5"]}),"\n"]}),"\n",(0,i.jsx)(s.p,{children:"\u9a8c\u8bc1"}),"\n",(0,i.jsx)(s.pre,{children:(0,i.jsx)(s.code,{children:"$ systemctl status k3s\n\u25cf k3s.service - Lightweight Kubernetes\n   Loaded: loaded (/etc/systemd/system/k3s.service; enabled; vendor preset: disabled)\n   Active: active (running) since Sun 2023-12-24 09:52:44 EST; 1min 4s ago\n     Docs: https://k3s.io\n  Process: 11405 ExecStartPre=/sbin/modprobe overlay (code=exited, status=0/SUCCESS)\n  Process: 11402 ExecStartPre=/sbin/modprobe br_netfilter (code=exited, status=0/SUCCESS)\n  Process: 11399 ExecStartPre=/bin/sh -xc ! /usr/bin/systemctl is-enabled --quiet nm-cloud-setup.service (code=exited, status=0/SUCCESS)\n Main PID: 11408 (k3s-server)\n    Tasks: 59\n   Memory: 486.0M\n   CGroup: /system.slice/k3s.service\n           \u251c\u250011408 /usr/local/bin/k3s server\n           \u2514\u250011462 containerd \n$ kubectl get pod -A\nNAMESPACE     NAME                                      READY   STATUS    RESTARTS   AGE\nkube-system   local-path-provisioner-84db5d44d9-ws8wq   1/1     Running   0          19s\nkube-system   metrics-server-67c658944b-xj6mr           1/1     Running   0          19s\nkube-system   coredns-6799fbcd5-6k2ff                   1/1     Running   0          19s\n"})}),"\n",(0,i.jsxs)(s.p,{children:["\u5982\u679cpod/coredns\u51fa\u73b0\u62a5\u9519",(0,i.jsx)(s.code,{children:"plugin/forward: no nameservers found"}),",\u7f16\u8f91configmap ",(0,i.jsx)(s.code,{children:"kubectl edit cm coredns -n kube-system"}),", \u5c06 ",(0,i.jsx)(s.code,{children:"forward . /etc/resolv.conf"}),"\u8fd9\u4e00\u884c\u6ce8\u91ca\u3002\u4fdd\u5b58\u9000\u51fa\u540e\uff0c\u5220\u9664pod/coredns"]}),"\n",(0,i.jsx)(s.h2,{id:"\u79c1\u6709\u955c\u50cf\u4ed3\u5e93\u914d\u7f6e",children:"\u79c1\u6709\u955c\u50cf\u4ed3\u5e93\u914d\u7f6e"}),"\n",(0,i.jsxs)(s.p,{children:["\u6211\u4eec\u53ef\u4ee5\u5c06\u5bb9\u5668\u914d\u7f6e\u4e3a\u8fde\u63a5\u5230\u79c1\u6709\u955c\u50cf\u4ed3\u5e93\uff0c\u5e76\u5728\u8282\u70b9\u4e0a\u4f7f\u7528\u79c1\u6709\u955c\u50cf\u4ed3\u5e93\u62c9\u53d6\u79c1\u6709\u955c\u50cf\u3002K3s\u5728\u542f\u52a8\u65f6\u4f1a\u68c0\u67e5",(0,i.jsx)(s.code,{children:"/etc/rancher/k3s/"}),"\u4e2d\u662f\u5426\u5b58\u5728",(0,i.jsx)(s.code,{children:"registries.yaml"}),"\u6587\u4ef6\uff0c\u5b58\u5728\u7684\u8bdd\u5728\u542f\u52a8\u5bb9\u5668\u7684\u65f6\u5019\u4f1a\u4f7f\u7528\u8be5\u6587\u4ef6\u4e2d\u5b9a\u4e49\u7684\u955c\u50cf\u4ed3\u5e93\u3002"]}),"\n",(0,i.jsx)(s.pre,{children:(0,i.jsx)(s.code,{children:'$ sudo vi /etc/rancher/k3s/registries.yaml\nmirrors:\n  docker.io:\n    endpoint:\n      - "http://127.0.0.1:5000"\n$ sudo systemctl restart k3s\n'})}),"\n",(0,i.jsx)(s.p,{children:"\u6bd4\u5982\u4e0a\u9762\u7684\u914d\u7f6e\uff0c\u5f53\u6211\u4eec\u5bb9\u5668\u9700\u8981\u7684\u955c\u50cf\u4ed3\u5e93\u662fdocker.io\u65f6\uff0ck3s\u4f1a\u4ece127.0.0.1:5000\u4ed3\u5e93\u6765\u62c9\u53d6\u955c\u50cf\u3002"})]})}function u(e={}){const{wrapper:s}={...(0,r.a)(),...e.components};return s?(0,i.jsx)(s,{...e,children:(0,i.jsx)(a,{...e})}):a(e)}},1151:(e,s,n)=>{n.d(s,{Z:()=>c,a:()=>t});var i=n(7294);const r={},l=i.createContext(r);function t(e){const s=i.useContext(l);return i.useMemo((function(){return"function"==typeof e?e(s):{...s,...e}}),[s,e])}function c(e){let s;return s=e.disableParentContext?"function"==typeof e.components?e.components(r):e.components||r:t(e.components),i.createElement(l.Provider,{value:s},e.children)}}}]);