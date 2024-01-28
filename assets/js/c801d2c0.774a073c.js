"use strict";(self.webpackChunkoldbone_site_docusaurus=self.webpackChunkoldbone_site_docusaurus||[]).push([[4389],{8036:(e,n,r)=>{r.r(n),r.d(n,{assets:()=>d,contentTitle:()=>i,default:()=>h,frontMatter:()=>c,metadata:()=>o,toc:()=>l});var s=r(5893),t=r(1151);const c={},i="\u5f00\u7bc7",o={id:"cloud-native/docker-concept/docker-intro",title:"\u5f00\u7bc7",description:"\u5bb9\u5668\u7684\u672c\u8d28",source:"@site/docs/cloud-native/docker-concept/0-docker-intro.md",sourceDirName:"cloud-native/docker-concept",slug:"/cloud-native/docker-concept/docker-intro",permalink:"/docs/cloud-native/docker-concept/docker-intro",draft:!1,unlisted:!1,tags:[],version:"current",sidebarPosition:0,frontMatter:{},sidebar:"cloudNativeSidebar",previous:{title:"5\u5206\u949fDocker\u5bb9\u5668\u539f\u7406",permalink:"/docs/category/5\u5206\u949fdocker\u5bb9\u5668\u539f\u7406"},next:{title:"namespace \u89c2\u5bdf\u548c\u5b9e\u8df5\u7bc7",permalink:"/docs/cloud-native/docker-concept/docker-namespace"}},d={},l=[{value:"\u5bb9\u5668\u7684\u672c\u8d28",id:"\u5bb9\u5668\u7684\u672c\u8d28",level:2},{value:"\u865a\u62df\u673a(VM)\u548c\u5bb9\u5668(Container)\u7684\u533a\u522b",id:"\u865a\u62df\u673avm\u548c\u5bb9\u5668container\u7684\u533a\u522b",level:2},{value:"\u5bb9\u5668\u7684\u9694\u79bb\u548c\u9650\u5236",id:"\u5bb9\u5668\u7684\u9694\u79bb\u548c\u9650\u5236",level:2},{value:"\u9694\u79bb",id:"\u9694\u79bb",level:3},{value:"\u9650\u5236",id:"\u9650\u5236",level:3},{value:"\u5bb9\u5668\u7684\u6587\u4ef6\u7cfb\u7edf",id:"\u5bb9\u5668\u7684\u6587\u4ef6\u7cfb\u7edf",level:2}];function a(e){const n={br:"br",code:"code",h1:"h1",h2:"h2",h3:"h3",img:"img",p:"p",strong:"strong",table:"table",tbody:"tbody",td:"td",th:"th",thead:"thead",tr:"tr",...(0,t.a)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(n.h1,{id:"\u5f00\u7bc7",children:"\u5f00\u7bc7"}),"\n",(0,s.jsx)(n.h2,{id:"\u5bb9\u5668\u7684\u672c\u8d28",children:"\u5bb9\u5668\u7684\u672c\u8d28"}),"\n",(0,s.jsxs)(n.p,{children:["\u5bb9\u5668\u6280\u672f\uff0c\u662f\u4e00\u79cd",(0,s.jsx)(n.strong,{children:"\u6c99\u76d2"}),"\u6280\u672f\uff0c \u8fd9\u79cd\u6280\u672f\u5c31\u50cf",(0,s.jsx)(n.strong,{children:"\u96c6\u88c5\u7bb1"}),"\u4e00\u6837\uff0c\u628a\u4f60\u7684\u5e94\u7528\u88c5\u5230\u91cc\u9762\u53bb\uff0c\u56e0\u6b64\u5e94\u7528\u4e0e\u5e94\u7528\u4e4b\u95f4\u56e0\u4e3a\u6709\u4e86",(0,s.jsx)(n.strong,{children:"\u96c6\u88c5\u7bb1"}),"\u7684\u5b58\u5728\uff0c\u4e0d\u4f1a\u76f8\u4e92\u5e72\u6270\u3002\u800c\u4e14\u88ab\u88c5\u8fdb",(0,s.jsx)(n.strong,{children:"\u96c6\u88c5\u7bb1"}),"\u7684\u5e94\u7528\uff0c\u4e5f\u53ef\u4ee5\u5f88\u65b9\u4fbf\u5730\u642c\u6765\u642c\u53bb\u3002",(0,s.jsx)(n.br,{}),"\n","\u5bb9\u5668\u6280\u672f\u7684\u6838\u5fc3\u529f\u80fd\uff0c\u5c31\u662f\u63d0\u4f9b\u8fd9\u4e48\u4e00\u4e2a",(0,s.jsx)(n.strong,{children:"\u96c6\u88c5\u7bb1"}),"\uff0c\u4e00\u4e2a",(0,s.jsx)(n.strong,{children:"\u88ab\u7ea6\u675f\u597d\u8fb9\u754c\u548c\u6761\u4ef6\u7684\u8fdb\u7a0b"}),"\u3002",(0,s.jsx)(n.br,{}),"\n","\u5bb9\u5668\uff0c\u5176\u5b9e\u5c31\u662f\u4e00\u79cd\u7279\u6b8a\u7684\u8fdb\u7a0b\u3002"]}),"\n",(0,s.jsx)(n.h2,{id:"\u865a\u62df\u673avm\u548c\u5bb9\u5668container\u7684\u533a\u522b",children:"\u865a\u62df\u673a(VM)\u548c\u5bb9\u5668(Container)\u7684\u533a\u522b"}),"\n",(0,s.jsxs)(n.table,{children:[(0,s.jsx)(n.thead,{children:(0,s.jsxs)(n.tr,{children:[(0,s.jsx)(n.th,{style:{textAlign:"center"},children:"VM"}),(0,s.jsx)(n.th,{style:{textAlign:"center"},children:"Container"})]})}),(0,s.jsx)(n.tbody,{children:(0,s.jsxs)(n.tr,{children:[(0,s.jsx)(n.td,{style:{textAlign:"center"},children:(0,s.jsx)(n.img,{alt:"vm.png",src:r(9829).Z+"",width:"429",height:"369"})}),(0,s.jsx)(n.td,{style:{textAlign:"center"},children:(0,s.jsx)(n.img,{alt:"container.png",src:r(3588).Z+"",width:"428",height:"367"})})]})})]}),"\n",(0,s.jsxs)(n.p,{children:["\u865a\u62df\u673a\u7684\u8fd0\u884c\uff0c\u9700\u8981",(0,s.jsx)(n.strong,{children:"Hypervisor"}),"\u4e3a\u5176\u521b\u5efa\u865a\u62df\u786c\u4ef6/\u56fa\u4ef6\uff0c\u800c\u4e14\u8fd8\u9700\u8981\u642d\u5efa",(0,s.jsx)(n.strong,{children:"Guest OS"}),"\u624d\u80fd\u8fd0\u884c\u7528\u6237\u7684\u5e94\u7528\u8fdb\u7a0b\u3002\u8fd9\u4e2a\u865a\u62df\u673a\u662f\u771f\u5b9e\u5b58\u5728\u7684\uff0c\u65e0\u8bba\u7528\u6237\u7684\u5e94\u7528\u8fdb\u7a0b\u662f\u5426\u8fd0\u884c\uff0c\u90fd\u9700\u8981\u6d88\u8017\u548c\u5360\u7528\u4e00\u5b9a\u7684\u8d44\u6e90\u6765\u8fd0\u884c\u865a\u62df\u673a\u3002\u800c\u4e14\u5e94\u7528\u8fdb\u7a0b\u5728\u865a\u62df\u673a\u4e0a\u9762\u8fd0\u884c\u3001\u8c03\u7528\u8d44\u6e90\u7684\u65f6\u5019\uff0c\u65e0\u53ef\u907f\u514d\u5730\u90fd\u8981\u7ecf\u8fc7\u865a\u62df\u673a\u7684\u8fde\u63a5\u548c\u5904\u7406\uff0c\u5e26\u6765\u989d\u5916\u7684\u6027\u80fd\u635f\u8017\u3002",(0,s.jsx)(n.br,{}),"\n","\u5bb9\u5668\u5316\u540e\u7684\u7528\u6237\u5e94\u7528\uff0c\u5b9e\u9645\u4e0a\u8fd8\u662f\u4e00\u4e2a\u8fdb\u7a0b\uff0c\u5e76\u4e0d\u4f1a\u6709\u865a\u62df\u5316\u5e26\u6765\u7684\u6027\u80fd\u635f\u8017\u3002\u800c\u4e14\u5bb9\u5668\u672c\u8eab\u7684",(0,s.jsx)(n.strong,{children:"\u9694\u79bb"}),"\uff0c\u662f\u901a\u8fc7",(0,s.jsx)(n.code,{children:"namespace"}),"\u6280\u672f\u6765\u5b9e\u73b0\u7684\uff0c\u8d44\u6e90\u7684\u6d88\u8017\u548c\u5360\u7528\u53ef\u4ee5\u5ffd\u7565\u4e0d\u8ba1\u3002"]}),"\n",(0,s.jsx)(n.h2,{id:"\u5bb9\u5668\u7684\u9694\u79bb\u548c\u9650\u5236",children:"\u5bb9\u5668\u7684\u9694\u79bb\u548c\u9650\u5236"}),"\n",(0,s.jsxs)(n.p,{children:["\u5bf9\u4e8eDocker\u7b49\u5927\u591a\u6570Linux\u5bb9\u5668\u6765\u8bf4\uff0c",(0,s.jsx)(n.strong,{children:"\u9694\u79bb"}),"\u662f\u901a\u8fc7",(0,s.jsx)(n.strong,{children:"namespace"}),"\u6280\u672f\u6765\u5b9e\u73b0\uff0c\u800c",(0,s.jsx)(n.strong,{children:"\u9650\u5236"}),"\u662f\u901a\u8fc7",(0,s.jsx)(n.strong,{children:"cgroup"}),"\u6280\u672f\u6765\u63a7\u5236\u3002"]}),"\n",(0,s.jsx)(n.h3,{id:"\u9694\u79bb",children:"\u9694\u79bb"}),"\n",(0,s.jsxs)(n.p,{children:["\u8d44\u6e90\u9694\u79bb\u3002\u901a\u8fc7",(0,s.jsx)(n.strong,{children:"Linux namespace"}),"\u6280\u672f\u53ef\u4ee5\u5bf9\u5168\u5c40\u7cfb\u7edf\u8d44\u6e90\u8fdb\u884c\u5c01\u88c5\u548c\u9694\u79bb\uff0c\u8ba9\u5bb9\u5668\u5185\u7684\u8fdb\u7a0b\u53ea\u80fd\u770b\u5230\u81ea\u5df1",(0,s.jsx)(n.strong,{children:"namespace"}),"\u7684\u76f8\u5173\u8d44\u6e90\uff0c\u8fd9\u6837\u5c31\u548c\u5176\u4ed6",(0,s.jsx)(n.strong,{children:"namespace"}),"\u7684\u8fdb\u7a0b\u8d77\u5230\u4e86\u9694\u79bb\u7684\u4f5c\u7528\u3002\u6539\u53d8\u4e00\u4e2a",(0,s.jsx)(n.strong,{children:"namespace"}),"\u91cc\u9762\u7684\u8d44\u6e90\uff0c\u53ea\u4f1a\u5f71\u54cd\u540c\u4e00\u4e2a",(0,s.jsx)(n.strong,{children:"namespace"}),"\u91cc\u7684\u8fdb\u7a0b\uff0c\u5bf9\u5176\u4ed6",(0,s.jsx)(n.strong,{children:"namespace"}),"\u7684\u8fdb\u7a0b\u6ca1\u6709\u5f71\u54cd\u3002",(0,s.jsx)(n.br,{}),"\n","\u5728",(0,s.jsx)(n.strong,{children:"Linux namespace"}),"\u4e2d\uff0c\u867d\u7136\u80fd\u591f\u9694\u79bb\u5f88\u591a\u8d44\u6e90\uff0c\u5305\u62ec\u6027\u80fd\u8d44\u6e90\u3001\u6587\u4ef6\u3001\u8bbe\u5907\u3001\u72b6\u6001\u3001\u914d\u7f6e\u7b49\uff0c\u4f46\u4e5f\u6709\u4e0d\u80fd\u9694\u79bb\u7684\u5bf9\u8c61\u3002\u6700\u5178\u578b\u7684\u662f\u65f6\u95f4\uff0c\u5f53\u4f60\u4fee\u6539\u5bb9\u5668\u91cc\u7684\u65f6\u95f4\u65f6\uff0c\u540c\u65f6\u4f1a\u5bf9\u5bbf\u4e3b\u673a\u7684\u65f6\u95f4\u9020\u6210\u4fee\u6539\u3002"]}),"\n",(0,s.jsx)(n.h3,{id:"\u9650\u5236",children:"\u9650\u5236"}),"\n",(0,s.jsxs)(n.p,{children:["Docker\u5bb9\u5668\u4e2d\u7684\u8fdb\u7a0b\uff0c\u867d\u7136\u5728",(0,s.jsx)(n.strong,{children:"Linux namespace"}),"\u6280\u672f\u7684\u9694\u79bb\u4e0b\uff0c\u5404\u81ea\u4e4b\u95f4\u7684\u8fd0\u884c\u4e0d\u53d7\u5f71\u54cd\uff0c\u4f46\u662f\u4ed6\u4eec\u5b9e\u9645\u4e0a\u8fd8\u662f\u5bbf\u4e3b\u673a\u4e0a\u9762\u7684\u4e00\u4e2a\u8fdb\u7a0b\uff0c\u6d88\u8017\u548c\u5360\u7528\u7740\u5bbf\u4e3b\u673a\u7684\u8d44\u6e90\u3002\u5bbf\u4e3b\u673a\u7684\u8d44\u6e90\u662f\u6709\u9650\u4e14\u516c\u7528\u7684\uff0c\u5982\u679c\u4e0d\u5bf9\u8fd9\u4e9b\u8fdb\u7a0b\u8fdb\u884c\u8d44\u6e90\u7684\u9650\u5236\uff0c\u8fd9\u4e9b\u8fdb\u7a0b\u53ef\u80fd\u4f1a\u5360\u7528\u5f88\u591a\u7684\u7cfb\u7edf\u8d44\u6e90\uff0c\u4ece\u800c\u5f71\u54cd\u5230\u5176\u4ed6\u8fdb\u7a0b\u7684\u6b63\u5e38\u8fd0\u884c\u3002",(0,s.jsx)(n.br,{}),"\n",(0,s.jsx)(n.strong,{children:"Linux cgroups"}),"\u6280\u672f\u662fLinux\u5185\u6838\u4e2d\u7528\u6765\u4e3a\u8fdb\u7a0b\u9650\u5236\u8d44\u6e90\u8c03\u7528\u7684\u4e00\u4e2a\u91cd\u8981\u6280\u672f\u3002",(0,s.jsx)(n.strong,{children:"Linux cgroups"}),"\u7684\u5168\u79f0\u662f",(0,s.jsx)(n.strong,{children:"Linux Control Group"}),", \u5b83\u6700\u4e3b\u8981\u7684\u529f\u80fd\uff0c\u5c31\u662f\u9650\u5236\u4e00\u4e2a\u8fdb\u7a0b\u80fd\u591f\u4f7f\u7528\u7684\u8d44\u6e90\u4e0a\u9650\uff0c\u5305\u62ec\u5185\u5b58\u3001CPU\u3001\u78c1\u76d8\u7a7a\u95f4\u3001\u7f51\u7edc\u5e26\u5bbd\u7b49\u3002"]}),"\n",(0,s.jsx)(n.h2,{id:"\u5bb9\u5668\u7684\u6587\u4ef6\u7cfb\u7edf",children:"\u5bb9\u5668\u7684\u6587\u4ef6\u7cfb\u7edf"}),"\n",(0,s.jsxs)(n.p,{children:["\u4e3a\u4e86\u8ba9\u5bb9\u5668\u5185\u90e8",(0,s.jsx)(n.strong,{children:"\u66f4\u50cf"}),"\u4e00\u4e2a\u72ec\u7acb\u7684\u7cfb\u7edf\uff0c\u4e00\u822c\u6211\u4eec\u4f1a\u628a\u4e00\u6574\u4e2a\u64cd\u4f5c\u7cfb\u7edf\u7684\u6587\u4ef6\u7cfb\u7edf\uff0c\u6302\u8f7d\u5230\u5bb9\u5668\u7684\u6839\u76ee\u5f55",(0,s.jsx)(n.code,{children:"/"}),"\u4e0b\u9762\uff0c\u7528\u6765\u7ed9\u5bb9\u5668\u5185\u90e8\u7684\u8fdb\u7a0b\u63d0\u4f9b\u9694\u79bb\u540e\u7684\u6267\u884c\u73af\u5883\u3002\u8fd9\u4e2a\u6587\u4ef6\u7cfb\u7edf\uff0c\u5c31\u662f",(0,s.jsx)(n.strong,{children:"\u5bb9\u5668\u955c\u50cf"}),"\uff0c\u4e5f\u53eb\u505a",(0,s.jsx)(n.strong,{children:"rootfs"}),"."]})]})}function h(e={}){const{wrapper:n}={...(0,t.a)(),...e.components};return n?(0,s.jsx)(n,{...e,children:(0,s.jsx)(a,{...e})}):a(e)}},3588:(e,n,r)=>{r.d(n,{Z:()=>s});const s=r.p+"assets/images/container-466b6528a0eac7c3da54fd8b8080d634.png"},9829:(e,n,r)=>{r.d(n,{Z:()=>s});const s=r.p+"assets/images/vm-6ad212205b90c4d1358f1b6651516ddd.png"},1151:(e,n,r)=>{r.d(n,{Z:()=>o,a:()=>i});var s=r(7294);const t={},c=s.createContext(t);function i(e){const n=s.useContext(c);return s.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function o(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(t):e.components||t:i(e.components),s.createElement(c.Provider,{value:n},e.children)}}}]);