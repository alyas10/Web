function toggleSidebar() {
  const sidebar = document.getElementById("mySidebar");
  const main = document.getElementById("main");
  const openBtn = document.getElementById("openSidebarBtn");
  const appContainer = document.querySelector(".app-container");

  const isCollapsed = appContainer.classList.contains('sidebar-collapsed');
  //if (sidebar.style.visibility === "hidden" || sidebar.offsetWidth <= 60) {
   if(isCollapsed){
    // Развернуть
   /* sidebar.style.width = "250px";
    main.style.marginLeft = "250px";
    sidebar.style.visibility = "visible";
    sidebar.style.overflow = "auto";
    openBtn.style.display = "none";*/
    appContainer.classList.remove("sidebar-collapsed");
    openBtn.style.display = "none";
  } else {
    // Свернуть
    /*sidebar.style.width = "0";
    sidebar.style.visibility = "hidden";
    sidebar.style.overflow = "hidden";
    main.style.marginLeft = "0";
    openBtn.style.display = "flex";*/
    appContainer.classList.add("sidebar-collapsed");
    openBtn.style.display = "flex";
  }
}