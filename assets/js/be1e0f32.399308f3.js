"use strict";(self.webpackChunkoldbone_site_docusaurus=self.webpackChunkoldbone_site_docusaurus||[]).push([[5396],{470:(e,n,r)=>{r.r(n),r.d(n,{assets:()=>o,contentTitle:()=>c,default:()=>u,frontMatter:()=>d,metadata:()=>i,toc:()=>a});var t=r(5893),s=r(1151);const d={},c="Docker",i={id:"offline-install/docker",title:"Docker",description:"\u672c\u6587\u5185\u5bb9",source:"@site/docs/offline-install/1-docker.md",sourceDirName:"offline-install",slug:"/offline-install/docker",permalink:"/docs/offline-install/docker",draft:!1,unlisted:!1,tags:[],version:"current",sidebarPosition:1,frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"\u79bb\u7ebf\u73af\u5883\u5b89\u88c5",permalink:"/docs/category/\u79bb\u7ebf\u73af\u5883\u5b89\u88c5"},next:{title:"k3s",permalink:"/docs/offline-install/k3s"}},o={},a=[{value:"\u672c\u6587\u5185\u5bb9",id:"\u672c\u6587\u5185\u5bb9",level:2},{value:"\u80cc\u666f\u63cf\u8ff0",id:"\u80cc\u666f\u63cf\u8ff0",level:2},{value:"\u5b89\u88c5docker",id:"\u5b89\u88c5docker",level:2},{value:"\u642d\u5efadocker\u79c1\u6709\u955c\u50cf\u4ed3\u5e93",id:"\u642d\u5efadocker\u79c1\u6709\u955c\u50cf\u4ed3\u5e93",level:2}];function l(e){const n={a:"a",blockquote:"blockquote",br:"br",code:"code",h1:"h1",h2:"h2",li:"li",p:"p",pre:"pre",ul:"ul",...(0,s.a)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(n.h1,{id:"docker",children:"Docker"}),"\n",(0,t.jsx)(n.h2,{id:"\u672c\u6587\u5185\u5bb9",children:"\u672c\u6587\u5185\u5bb9"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"\u901a\u8fc7binary\u65b9\u5f0f\u5b89\u88c5docker\uff0c\u5e76\u6ce8\u518c\u6210\u670d\u52a1"}),"\n",(0,t.jsx)(n.li,{children:"\u642d\u5efadocker\u79c1\u6709\u955c\u50cf\u4ed3\u5e93"}),"\n"]}),"\n",(0,t.jsx)(n.h2,{id:"\u80cc\u666f\u63cf\u8ff0",children:"\u80cc\u666f\u63cf\u8ff0"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"\u6240\u5728\u7684\u670d\u52a1\u5668\u662f\u4e00\u4e2a\u5185\u7f51\uff0c\u65e0\u6cd5\u8bbf\u95ee\u5230\u5916\u7f51\u73af\u5883\u3002"}),"\n",(0,t.jsx)(n.li,{children:"\u670d\u52a1\u5668OS\u7248\u672c\u662f CentOS 7.9"}),"\n"]}),"\n",(0,t.jsx)(n.h2,{id:"\u5b89\u88c5docker",children:"\u5b89\u88c5docker"}),"\n",(0,t.jsxs)(n.p,{children:["\u83b7\u53d6\u6240\u9700\u7248\u672c\u7684docker binary\u5305\uff0c\u5b98\u65b9\u94fe\u63a5\u5728 ",(0,t.jsx)(n.a,{href:"https://download.docker.com/linux/static/stable/x86_64/",children:"https://download.docker.com/linux/static/stable/x86_64/"}),". \u5728\u8fd9\u91cc\u6211\u9009\u62e9\u4e86\u6700\u65b0\u7248\u672c\u7684",(0,t.jsx)(n.code,{children:" docker-24.0.7.tgz"})]}),"\n",(0,t.jsx)(n.p,{children:"\u5c06\u538b\u7f29\u5305\u4e0a\u4f20\u5230\u76ee\u6807\u670d\u52a1\u5668\u3002"}),"\n",(0,t.jsxs)(n.p,{children:["\u89e3\u538b\u538b\u7f29\u5305\uff0c\u5e76\u628a\u6587\u4ef6\u653e\u5728",(0,t.jsx)(n.code,{children:"/usr/bin/"})]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{children:"$ tar zxvf docker-24.0.7.tgz \ndocker/\ndocker/docker\ndocker/docker-init\ndocker/dockerd\ndocker/runc\ndocker/ctr\ndocker/containerd-shim-runc-v2\ndocker/containerd\ndocker/docker-proxy\n$ sudo cp docker/* /usr/bin/\n"})}),"\n",(0,t.jsx)(n.p,{children:"\u7f16\u5199service\u6587\u4ef6"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{children:"$ vim /etc/systemd/system/docker.service\n[Unit]\nDescription=Docker Application Container Engine\nDocumentation=https://docs.docker.com\nAfter=network-online.target firewalld.service\nWants=network-online.target\n[Service]\nType=notify\n## the default is not to use systemd for cgroups because the delegate issues still\n## exists and systemd currently does not support the cgroup feature set required\n## for containers run by docker\nExecStart=/usr/bin/dockerd\nExecReload=/bin/kill -s HUP $MAINPID\n## Having non-zero Limit*s causes performance problems due to accounting overhead\n## in the kernel. We recommend using cgroups to do container-local accounting.\nLimitNOFILE=infinity\nLimitNPROC=infinity\nLimitCORE=infinity\n## Uncomment TasksMax if your systemd version supports it.\n## Only systemd 226 and above support this version.\n#TasksMax=infinity\nTimeoutStartSec=0\n## set delegate yes so that systemd does not reset the cgroups of docker containers\nDelegate=yes\n## kill only the docker process, not all processes in the cgroup\nKillMode=process\n## restart the docker process if it exits prematurely\nRestart=on-failure\nStartLimitBurst=3\nStartLimitInterval=60s\n \n[Install]\nWantedBy=multi-user.target\n"})}),"\n",(0,t.jsx)(n.p,{children:"\u65b0\u589e\u7528\u6237\u7ec4 docker\u3002\u5176\u4ed6\u7528\u6237\u9700\u8981\u6709docker\u7684\u6267\u884c\u6743\u9650\uff0c\u53ea\u9700\u5c06\u7528\u6237\u52a0\u5165docker\u7528\u6237\u7ec4\u5373\u53ef"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{children:"$ sudo groupadd docker\n$ sudo usermod -aG docker $USER\n"})}),"\n",(0,t.jsx)(n.p,{children:"\u542f\u52a8docker service,\u5e76\u8bbe\u7f6e\u5f00\u673a\u81ea\u542f\u52a8"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{children:"$ sudo chmod +x /etc/systemd/system/docker.service\n$ sudo systemctl daemon-reload\n$ sudo systemctl start docker\n$ sudo systemctl enable docker\n"})}),"\n",(0,t.jsx)(n.p,{children:"\u9a8c\u8bc1"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{children:"$ systemctl status docker                                                         \n\u25cf docker.service - Docker Application Container Engine\n   Loaded: loaded (/etc/systemd/system/docker.service; enabled; vendor preset: disabled)\n   Active: active (running) since Sun 2023-12-24 02:13:06 EST; 1min 1s ago\n     Docs: https://docs.docker.com\n Main PID: 1764 (dockerd)\n   CGroup: /system.slice/docker.service\n           \u251c\u25001764 /usr/bin/dockerd\n           \u2514\u25001771 containerd --config /var/run/docker/containerd/containerd.toml\n$ docker -v\nDocker version 24.0.7, build afdd53b\n"})}),"\n",(0,t.jsx)(n.h2,{id:"\u642d\u5efadocker\u79c1\u6709\u955c\u50cf\u4ed3\u5e93",children:"\u642d\u5efadocker\u79c1\u6709\u955c\u50cf\u4ed3\u5e93"}),"\n",(0,t.jsx)(n.p,{children:"\u5728\u4e00\u53f0\u53ef\u4ee5\u8bbf\u95ee\u5230\u5916\u7f51\u7684\u670d\u52a1\u5668\u4e0a\u9762\u51c6\u5907registry\u955c\u50cf\u5305"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{children:"$ docker pull registry\nUsing default tag: latest\nlatest: Pulling from library/registry\nc926b61bad3b: Pull complete \n5501dced60f8: Pull complete \ne875fe5e6b9c: Pull complete \n21f4bf2f86f9: Pull complete \n98513cca25bb: Pull complete \nDigest: sha256:0a182cb82c93939407967d6d71d6caf11dcef0e5689c6afe2d60518e3b34ab86\nStatus: Downloaded newer image for registry:latest\ndocker.io/library/registry:latest\n$ docker save -o registry.tar registry\n$ gzip registry.tar\n"})}),"\n",(0,t.jsxs)(n.p,{children:["\u4e0a\u4f20\u955c\u50cf\u5305",(0,t.jsx)(n.code,{children:"registry.tar.gz"}),"\u5230\u76ee\u6807\u670d\u52a1\u5668"]}),"\n",(0,t.jsx)(n.p,{children:"\u89e3\u538b\u52a0\u8f7d\u955c\u50cf"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{children:"$ gunzip registry.tar.gz \n$ docker load < registry.tar \n4693057ce236: Loading layer [==================================================>]  7.626MB/7.626MB\nf4285c491509: Loading layer [==================================================>]  771.6kB/771.6kB\n90d6ca1e837f: Loading layer [==================================================>]   16.2MB/16.2MB\nf79c4d8837b6: Loading layer [==================================================>]  4.096kB/4.096kB\n85f82aceeda3: Loading layer [==================================================>]  2.048kB/2.048kB\nLoaded image: registry:2.8.2\n9fe9a137fd00: Loading layer [==================================================>]   7.63MB/7.63MB\nd9bce47b357e: Loading layer [==================================================>]  771.6kB/771.6kB\nafcdb1715fb3: Loading layer [==================================================>]  17.55MB/17.55MB\n9f383ae4f64d: Loading layer [==================================================>]  4.096kB/4.096kB\n645ddea72735: Loading layer [==================================================>]  2.048kB/2.048kB\nLoaded image: registry:latest\n"})}),"\n",(0,t.jsx)(n.p,{children:"\u542f\u52a8\u79c1\u6709\u955c\u50cf\u4ed3\u5e93\u5bb9\u5668"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{children:"$ sudo mkdir -p /data/registry\n$ docker run -itd -v /data/registry/:/docker/registry -p 5000:5000 --restart=always --name private-registry registry:latest\n98cb5c25871b94420418094efb19f36c45c7fbe18274229e9f8c4b00328ec180\n"})}),"\n",(0,t.jsx)(n.p,{children:"\u53c2\u6570\u8bf4\u660e"}),"\n",(0,t.jsxs)(n.blockquote,{children:["\n",(0,t.jsxs)(n.p,{children:["-itd\uff1a\u5728\u5bb9\u5668\u4e2d\u6253\u5f00\u4e00\u4e2a\u4f2a\u7ec8\u7aef\u8fdb\u884c\u4ea4\u4e92\u64cd\u4f5c\uff0c\u5e76\u5728\u540e\u53f0\u8fd0\u884c",(0,t.jsx)(n.br,{}),"\n","-v\uff1a\u6620\u5c04\u76ee\u5f55, \u5c06\u5bbf\u4e3b\u673a\u7684/data/registry \u6620\u5c04\u5230\u5bb9\u5668\u7684/dcoker/registry",(0,t.jsx)(n.br,{}),"\n","-p\uff1a\u6620\u5c04\u7aef\u53e3, \u5c06\u5bbf\u4e3b\u673a\u76845000\u7aef\u53e3\u6620\u5c04\u5230\u5bb9\u5668\u76845000\u7aef\u53e3"]}),"\n"]}),"\n",(0,t.jsx)(n.p,{children:"\u6d4b\u8bd5\u4e0a\u4f20\u955c\u50cf\u5230private-registry"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{children:"$ docker tag registry 127.0.0.1:5000/registry\n$ docker push 127.0.0.1:5000/registry\nUsing default tag: latest\nThe push refers to repository [127.0.0.1:5000/registry]\n645ddea72735: Pushed \n9f383ae4f64d: Pushed \nafcdb1715fb3: Pushed \nd9bce47b357e: Pushed \n9fe9a137fd00: Pushed \nlatest: digest: sha256:860f379a011eddfab604d9acfe3cf50b2d6e958026fb0f977132b0b083b1a3d7 size: 1363\n"})}),"\n",(0,t.jsx)(n.p,{children:"\u6d4b\u8bd5\u4e0b\u8f7d\u955c\u50cf"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{children:"$ docker rmi 127.0.0.1:5000/registry\nUntagged: 127.0.0.1:5000/registry:latest\nUntagged: 127.0.0.1:5000/registry@sha256:860f379a011eddfab604d9acfe3cf50b2d6e958026fb0f977132b0b083b1a3d7\n$ docker pull 127.0.0.1:5000/registry\nUsing default tag: latest\nlatest: Pulling from registry\nDigest: sha256:860f379a011eddfab604d9acfe3cf50b2d6e958026fb0f977132b0b083b1a3d7\nStatus: Downloaded newer image for 127.0.0.1:5000/registry:latest\n127.0.0.1:5000/registry:latest\n"})}),"\n",(0,t.jsx)(n.p,{children:"\u67e5\u770b\u79c1\u6709\u4ed3\u5e93\u955c\u50cf"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{children:'$ curl http://localhost:5000/v2/_catalog\n{"repositories":["registry"]}\n'})})]})}function u(e={}){const{wrapper:n}={...(0,s.a)(),...e.components};return n?(0,t.jsx)(n,{...e,children:(0,t.jsx)(l,{...e})}):l(e)}},1151:(e,n,r)=>{r.d(n,{Z:()=>i,a:()=>c});var t=r(7294);const s={},d=t.createContext(s);function c(e){const n=t.useContext(d);return t.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function i(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:c(e.components),t.createElement(d.Provider,{value:n},e.children)}}}]);